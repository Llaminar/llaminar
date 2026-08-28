/**
 * @file Test__MoEOverlayResidencyAuthority.cpp
 * @brief Device-free protocol tests for histogram-driven ExpertOverlay tiers.
 *
 * These tests prove the control-plane invariants required by the hardware
 * parity campaigns: hottest-first placement, capacity-preserving promotion and
 * demotion across hot/warm/cold domains, two-phase publication ordering,
 * ticket-epoch overlap, deferred shadow capacity, rollback, static immobility,
 * and PerfStats evidence.
 */

#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        class ScopedPerfStats final
        {
        public:
            ScopedPerfStats()
            {
                if (const char *old = std::getenv("LLAMINAR_PERF_STATS_JSON"))
                {
                    had_old_ = true;
                    old_value_ = old;
                }
                setenv("LLAMINAR_PERF_STATS_JSON", "1", 1);
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            ~ScopedPerfStats()
            {
                PerfStatsCollector::reset();
                if (had_old_)
                    setenv("LLAMINAR_PERF_STATS_JSON", old_value_.c_str(), 1);
                else
                    unsetenv("LLAMINAR_PERF_STATS_JSON");
                mutableDebugEnv().reload();
            }

        private:
            bool had_old_ = false;
            std::string old_value_;
        };

        RoutedExpertDomain domain(
            std::string name,
            GlobalDeviceAddress participant,
            int world_rank,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {participant};
            result.world_ranks = {world_rank};
            result.owner_rank = world_rank;
            result.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        RoutedExpertTier tier(
            std::string name,
            std::string domain_name,
            int priority,
            int capacity,
            bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = std::move(name);
            result.domain = std::move(domain_name);
            result.priority = priority;
            result.max_experts_per_layer = capacity;
            result.fallback = fallback;
            return result;
        }

        MoERoutedExpertPlacementPlan threeTierPlan(
            RoutedExpertResidencyPolicy policy,
            RoutedExpertOwnerOrder order)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = policy;
            plan.owner_order = order;
            plan.domains = {
                domain("cuda_hot", GlobalDeviceAddress::cuda(0, 0), 0,
                       CollectiveBackendType::NCCL),
                domain("rocm_warm", GlobalDeviceAddress::rocm(1, 0), 1,
                       CollectiveBackendType::RCCL),
                domain("cpu_cold", GlobalDeviceAddress::cpu(2), 2,
                       CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "cuda_hot", 0, 2),
                tier("warm", "rocm_warm", 1, 2),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /**
         * @brief One-domain ExpertOverlay over two CUDA participants.
         *
         * Keeping the declarative topology as `SingleDomain` is deliberate:
         * these tests prove it reaches the same epoch authority and same-tier
         * skew planner as a plan containing several priority tiers.
         */
        MoERoutedExpertPlacementPlan oneTierTwoParticipantPlan(
            RoutedExpertResidencyPolicy policy)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::SingleDomain;
            plan.continuation_domain = "cuda_domain";
            plan.shared_expert_domain = "cuda_domain";
            plan.residency_policy = policy;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;

            RoutedExpertDomain cuda_domain;
            cuda_domain.name = "cuda_domain";
            cuda_domain.scope = ExecutionDomainScope::RANK_LOCAL;
            cuda_domain.backend = CollectiveBackendType::NCCL;
            cuda_domain.participants = {
                GlobalDeviceAddress::cuda(0, 0),
                GlobalDeviceAddress::cuda(1, 0),
            };
            cuda_domain.owner_rank = 0;
            cuda_domain.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan.domains = {std::move(cuda_domain)};
            plan.routed_tiers = {
                tier("only_tier", "cuda_domain", 17, 0, true),
            };
            return plan;
        }

        /** @brief Histogram authority matching the one-tier participant set. */
        std::unique_ptr<DecodeExpertHistogram> oneTierHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 6;
            config.top_k = 1;
            config.window_size = 4;
            config.sockets = {DeviceId::cuda(0), DeviceId::cuda(1)};
            config.ownership = MoELayeredExpertOwnership::uniform(
                1, 2, {0, 0, 0, 1, 1, 1});
            return std::make_unique<DecodeExpertHistogram>(config);
        }

        /** @brief Histogram authority for two host-owned CPU participants. */
        std::unique_ptr<DecodeExpertHistogram> oneTierCpuHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 6;
            config.top_k = 1;
            config.window_size = 4;
            config.sockets = {DeviceId::cpu(), DeviceId::cpu()};
            config.ownership = MoELayeredExpertOwnership::uniform(
                1, 2, {0, 0, 0, 1, 1, 1});
            return std::make_unique<DecodeExpertHistogram>(config);
        }

        MoERoutedExpertModelMetadata modelMetadata()
        {
            MoERoutedExpertModelMetadata metadata;
            metadata.num_layers = 1;
            metadata.num_experts = 6;
            metadata.d_model = 16;
            metadata.routed_intermediate_size = 8;
            metadata.routed_quant_type = "F32";
            return metadata;
        }

        MoERoutedExpertPlacementPlan fourTierCyclePlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "tier_0";
            plan.shared_expert_domain = "tier_0";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                domain("tier_0", GlobalDeviceAddress::cuda(0, 0), 0,
                       CollectiveBackendType::NCCL),
                domain("tier_1", GlobalDeviceAddress::rocm(1, 0), 1,
                       CollectiveBackendType::RCCL),
                domain("tier_2", GlobalDeviceAddress::cpu(2), 2,
                       CollectiveBackendType::MPI),
                domain("tier_3", GlobalDeviceAddress::cpu(3), 3,
                       CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("tier_0", "tier_0", 0, 1),
                tier("tier_1", "tier_1", 1, 1),
                tier("tier_2", "tier_2", 2, 1),
                tier("tier_3", "tier_3", 3, 0, true),
            };
            return plan;
        }

        MoERoutedExpertModelMetadata fourTierMetadata()
        {
            auto metadata = modelMetadata();
            metadata.num_experts = 4;
            return metadata;
        }

        /** @brief Two GPU tiers plus one apportioned two-rank CPU cold tier. */
        MoERoutedExpertPlacementPlan nodeLocalThreeTierPlan(
            RoutedExpertOwnerOrder owner_order)
        {
            auto plan = fourTierCyclePlan();
            plan.owner_order = owner_order;
            plan.domains.resize(2);
            RoutedExpertDomain cold;
            cold.name = "cpu_cold";
            cold.scope = ExecutionDomainScope::NODE_LOCAL;
            cold.backend = CollectiveBackendType::UPI;
            cold.participants = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            cold.world_ranks = {0, 1};
            cold.owner_rank = 0;
            cold.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan.domains.push_back(std::move(cold));
            plan.routed_tiers = {
                tier("hot", "tier_0", 0, 1),
                tier("warm", "tier_1", 1, 1),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            plan.placements.clear();
            return plan;
        }

        /** @brief One accelerator tier above a two-participant CPU tier. */
        MoERoutedExpertPlacementPlan twoTierNodeLocalPlan(
            RoutedExpertOwnerOrder owner_order,
            int accelerator_capacity = 2)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "accelerator";
            plan.shared_expert_domain = "accelerator";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = owner_order;
            plan.domains = {
                domain(
                    "accelerator",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
            };

            RoutedExpertDomain cpu;
            cpu.name = "cpu_nodelocal";
            cpu.scope = ExecutionDomainScope::NODE_LOCAL;
            cpu.backend = CollectiveBackendType::UPI;
            cpu.participants = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            cpu.world_ranks = {0, 1};
            cpu.owner_rank = 0;
            cpu.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan.domains.push_back(std::move(cpu));
            plan.routed_tiers = {
                tier(
                    "priority_0",
                    "accelerator",
                    0,
                    accelerator_capacity),
                tier("priority_1", "cpu_nodelocal", 1, 0, true),
            };
            return plan;
        }

        /**
         * @brief One accelerator above three independently apportioned CPUs.
         *
         * Four participants let a tier exchange and a disjoint within-tier
         * ownership swap form separate closed cycles. That geometry is needed
         * to prove bounded scheduling fairness between the two Dynamic axes.
         */
        MoERoutedExpertPlacementPlan twoTierThreeCpuParticipantPlan(
            int accelerator_capacity = 2)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "accelerator";
            plan.shared_expert_domain = "accelerator";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                domain(
                    "accelerator",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
            };

            RoutedExpertDomain cpu;
            cpu.name = "cpu_nodelocal";
            cpu.scope = ExecutionDomainScope::NODE_LOCAL;
            cpu.backend = CollectiveBackendType::UPI;
            cpu.participants = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
                GlobalDeviceAddress::cpu(2),
            };
            cpu.world_ranks = {0, 1, 2};
            cpu.owner_rank = 0;
            cpu.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            plan.domains.push_back(std::move(cpu));
            plan.routed_tiers = {
                tier(
                    "priority_0",
                    "accelerator",
                    0,
                    accelerator_capacity),
                tier("priority_1", "cpu_nodelocal", 1, 0, true),
            };
            return plan;
        }

        /** @brief Geometry used to force two independently bounded swaps. */
        MoERoutedExpertModelMetadata eightExpertMetadata()
        {
            auto metadata = modelMetadata();
            metadata.num_experts = 8;
            return metadata;
        }

        /** @brief Geometry whose bounded target retains live CPU demand. */
        MoERoutedExpertModelMetadata twelveExpertMetadata()
        {
            auto metadata = modelMetadata();
            metadata.num_experts = 12;
            return metadata;
        }

        /** @brief Force a hot/warm/cold rotation that also shifts a CPU owner. */
        std::unique_ptr<DecodeExpertHistogram>
        nodeLocalRotationHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 4;
            config.top_k = 1;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                4,
                {0, 1, 2, 3});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::vector<std::uint64_t> counts{70, 60, 50, 100};
            histogram->mergeLayerCounts(0, counts.data(), 4, false);
            return histogram;
        }

        std::unique_ptr<DecodeExpertHistogram> fourTierRotationHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 4;
            config.top_k = 1;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                4,
                {0, 1, 2, 3});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::vector<uint64_t> counts{90, 80, 70, 100};
            histogram->mergeLayerCounts(0, counts.data(), 4, false);
            return histogram;
        }

        std::unique_ptr<DecodeExpertHistogram> histogramWithCounts(
            const std::vector<uint64_t> &counts)
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = static_cast<int>(counts.size());
            config.top_k = 2;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            std::vector<int> owners(counts.size(), 0);
            for (size_t expert_id = 0; expert_id < counts.size(); ++expert_id)
            {
                owners[expert_id] = static_cast<int>(
                    expert_id * 3 / counts.size());
            }
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                3,
                std::move(owners));
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            histogram->mergeLayerCounts(
                0,
                counts.data(),
                static_cast<int>(counts.size()),
                false);
            return histogram;
        }

        /** @brief Histogram ownership matching the four-participant test plan. */
        std::unique_ptr<DecodeExpertHistogram> fourParticipantHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 8;
            config.top_k = 2;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::cpu(),
                DeviceId::cpu(),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                4,
                {0, 0, 1, 1, 2, 2, 3, 3});
            return std::make_unique<DecodeExpertHistogram>(config);
        }

        /** @brief Six accelerator and two three-expert CPU owner buckets. */
        std::unique_ptr<DecodeExpertHistogram>
        threeParticipantTwelveExpertHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 12;
            config.top_k = 2;
            config.window_size = 4;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::cpu(),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                3,
                {0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2});
            return std::make_unique<DecodeExpertHistogram>(config);
        }

        /** @brief Build one immutable decode-only evidence generation. */
        std::shared_ptr<const DecodeExpertHistogramWindow> frozenWindow(
            uint64_t generation,
            const std::vector<uint64_t> &counts)
        {
            auto window =
                std::make_shared<DecodeExpertHistogramWindow>();
            window->generation = generation;
            window->num_layers = 1;
            window->num_experts = static_cast<int>(counts.size());
            window->expert_counts = counts;
            window->source_expert_counts.assign(
                counts.size() * kExpertHistogramProductionSourceCount,
                0);
            std::copy(
                counts.begin(),
                counts.end(),
                window->source_expert_counts.begin());
            uint64_t activations = 0;
            for (const uint64_t count : counts)
                activations += count;
            window->token_count = activations;
            window->source_token_counts[0] = activations;
            if (!window->valid())
                throw std::logic_error("Test histogram window is invalid");
            return window;
        }

        /** @brief Build one immutable phase-pure production evidence window. */
        std::shared_ptr<const DecodeExpertHistogramWindow> phasedFrozenWindow(
            uint64_t generation,
            const std::array<std::vector<uint64_t>,
                             kExpertHistogramProductionSourceCount> &counts)
        {
            const std::size_t experts = counts.front().size();
            if (experts == 0u || std::any_of(
                    counts.begin(),
                    counts.end(),
                    [experts](const auto &phase)
                    { return phase.size() != experts; }))
            {
                throw std::invalid_argument(
                    "Test phase histogram geometry differs");
            }

            auto window = std::make_shared<DecodeExpertHistogramWindow>();
            window->generation = generation;
            window->num_layers = 1;
            window->num_experts = static_cast<int>(experts);
            window->expert_counts.assign(experts, 0u);
            window->source_expert_counts.assign(
                experts * kExpertHistogramProductionSourceCount,
                0u);
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                for (std::size_t expert = 0; expert < experts; ++expert)
                {
                    const uint64_t value = counts[phase][expert];
                    window->source_expert_counts[
                        phase * experts + expert] = value;
                    window->expert_counts[expert] += value;
                    window->source_token_counts[phase] += value;
                    window->token_count += value;
                }
            }
            if (!window->valid())
                throw std::logic_error("Test phase histogram window is invalid");
            return window;
        }

        /** @brief Monotonic three-tier service profile for every phase. */
        std::shared_ptr<const MoERoutedTierServiceProfile>
        threeTierServiceProfile()
        {
            auto profile =
                std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "three-tier-service-v1";
            profile->costs = {
                {.tier_index = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
                {.tier_index = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {50, 100, 150}},
                {.tier_index = 2,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 200, 300}},
            };
            profile->participant_costs = {
                {.participant_id = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
                {.participant_id = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {50, 100, 150}},
                {.participant_id = 2,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 200, 300}},
            };
            return profile;
        }

        /** @brief Monotonic service costs for two integer-priority tiers. */
        std::shared_ptr<const MoERoutedTierServiceProfile>
        twoTierServiceProfile(int participant_count = 3)
        {
            auto profile =
                std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "two-priority-tier-service-v1";
            profile->costs = {
                {.tier_index = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
                {.tier_index = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 200, 300}},
            };
            for (int participant = 0;
                 participant < participant_count;
                 ++participant)
            {
                const std::array<uint64_t,
                                 kExpertHistogramProductionSourceCount>
                    cost = participant == 0
                               ? std::array<uint64_t,
                                            kExpertHistogramProductionSourceCount>{
                                     10, 20, 30}
                               : std::array<uint64_t,
                                            kExpertHistogramProductionSourceCount>{
                                     100, 200, 300};
                profile->participant_costs.push_back({
                    .participant_id = participant,
                    .layer = 0,
                    .nanoseconds_per_activation = cost,
                });
            }
            return profile;
        }

        /** @brief Measured service cost for a one-tier participant domain. */
        std::shared_ptr<const MoERoutedTierServiceProfile>
        oneTierServiceProfile()
        {
            auto profile =
                std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "one-tier-service-v1";
            profile->costs = {
                {.tier_index = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
            };
            profile->participant_costs = {
                {.participant_id = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
                {.participant_id = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {10, 20, 30}},
            };
            return profile;
        }

        /** @brief One tier whose second participant is materially slower. */
        std::shared_ptr<const MoERoutedTierServiceProfile>
        asymmetricOneTierServiceProfile()
        {
            auto profile =
                std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "one-tier-asymmetric-service-v1";
            profile->costs = {
                {.tier_index = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 100, 100}},
            };
            profile->participant_costs = {
                {.participant_id = 0,
                 .layer = 0,
                 .nanoseconds_per_activation = {1, 1, 1}},
                {.participant_id = 1,
                 .layer = 0,
                 .nanoseconds_per_activation = {100, 100, 100}},
            };
            return profile;
        }

        /** @brief Complete directed physical movement profile for three endpoints. */
        std::shared_ptr<const MoEOverlayMigrationCostProfile>
        threeParticipantMigrationProfile(
            uint64_t transfer_and_repack_ns,
            uint64_t inference_interference_ns)
        {
            auto profile =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            profile->identity = "three-participant-migration-v1";
            for (int source = 0; source < 3; ++source)
            {
                for (int destination = 0; destination < 3; ++destination)
                {
                    if (source == destination)
                        continue;
                    profile->costs.push_back({
                        .source_participant = source,
                        .destination_participant = destination,
                        .layer = 0,
                        .transfer_and_repack_ns = transfer_and_repack_ns,
                        .inference_interference_ns =
                            inference_interference_ns,
                    });
                }
            }
            return profile;
        }

        /** @brief Complete directed movement costs for four test endpoints. */
        std::shared_ptr<const MoEOverlayMigrationCostProfile>
        fourParticipantMigrationProfile(
            uint64_t transfer_and_repack_ns,
            uint64_t inference_interference_ns)
        {
            auto profile =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            profile->identity = "four-participant-migration-v1";
            for (int source = 0; source < 4; ++source)
            {
                for (int destination = 0; destination < 4; ++destination)
                {
                    if (source == destination)
                        continue;
                    profile->costs.push_back({
                        .source_participant = source,
                        .destination_participant = destination,
                        .layer = 0,
                        .transfer_and_repack_ns = transfer_and_repack_ns,
                        .inference_interference_ns =
                            inference_interference_ns,
                    });
                }
            }
            return profile;
        }

        /** @brief Complete directed movement costs for two same-domain GPUs. */
        std::shared_ptr<const MoEOverlayMigrationCostProfile>
        twoParticipantMigrationProfile()
        {
            auto profile =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            profile->identity = "two-participant-same-domain-v1";
            profile->costs = {
                {.source_participant = 0,
                 .destination_participant = 1,
                 .layer = 0,
                 .transfer_and_repack_ns = 1,
                 .inference_interference_ns = 1},
                {.source_participant = 1,
                 .destination_participant = 0,
                 .layer = 0,
                 .transfer_and_repack_ns = 1,
                 .inference_interference_ns = 1},
            };
            return profile;
        }

        class RecordingTransport final : public IMoEOverlayResidencyTransport
        {
        public:
            class Wave final : public IMoEOverlayResidencyWave
            {
            public:
                explicit Wave(RecordingTransport *owner) : owner_(owner) {}

                MoEOverlayResidencyWaveProgress pollStage(
                    std::string *error) noexcept override
                {
                    if (owner_->stage_pending_polls > 0)
                    {
                        --owner_->stage_pending_polls;
                        return MoEOverlayResidencyWaveProgress::Pending;
                    }
                    if (!owner_->stage_ok)
                    {
                        if (error)
                            *error = "injected stage failure";
                        return MoEOverlayResidencyWaveProgress::Failed;
                    }
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                bool beginPrepare(std::string *error) noexcept override
                {
                    owner_->calls.push_back("prepare");
                    owner_->epoch_seen_during_prepare =
                        owner_->authority_->snapshot()->epoch;
                    if (!owner_->prepare_ok && error)
                        *error = "injected preparation failure";
                    return owner_->prepare_ok;
                }

                MoEOverlayResidencyWaveProgress pollPrepare(
                    std::string *) noexcept override
                {
                    if (owner_->prepare_pending_polls > 0)
                    {
                        --owner_->prepare_pending_polls;
                        return MoEOverlayResidencyWaveProgress::Pending;
                    }
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                bool beginPublication(std::string *error) noexcept override
                {
                    owner_->calls.push_back("publish");
                    owner_->epoch_seen_during_publication =
                        owner_->authority_->snapshot()->epoch;

                    /*
                     * Selector publication may expose E+1 before public host
                     * admission advances. The exact-addressed candidate must
                     * therefore already be live at this irreversible edge.
                     */
                    const auto candidate_epoch =
                        owner_->epoch_seen_during_publication + 1u;
                    auto candidate = owner_->authority_->tryAcquireTicketSnapshot(
                        candidate_epoch);
                    owner_->candidate_was_exact_addressable =
                        candidate.has_value() &&
                        (*candidate)->epoch == candidate_epoch;

                    if (!owner_->publication_ok && error)
                        *error = "injected publication failure";
                    return owner_->publication_ok;
                }

                MoEOverlayResidencyWaveProgress pollPublication(
                    std::string *) noexcept override
                {
                    if (owner_->publication_pending_polls > 0)
                    {
                        --owner_->publication_pending_polls;
                        return MoEOverlayResidencyWaveProgress::Pending;
                    }
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                void abortStaged() noexcept override
                {
                    owner_->calls.push_back("abort");
                }

                void retirePrevious() noexcept override
                {
                    owner_->calls.push_back("retire");
                    owner_->epoch_seen_during_retire =
                        owner_->authority_->snapshot()->epoch;
                }

                void markAuthorityPublished() noexcept override
                {
                    owner_->calls.push_back("authority-published");
                }

                /** @brief Model independent device and host retirement edges. */
                MoEOverlayRetirementFenceProgress pollRetirementFence(
                    MoEOverlayLocalRetirementState local_state,
                    std::string *error) noexcept override
                {
                    ++owner_->retirement_fence_polls;
                    if (local_state.admission ==
                        MoEOverlayRetirementAdmissionState::Open)
                    {
                        if (owner_->retirement_admission_pending_polls > 0)
                        {
                            --owner_->retirement_admission_pending_polls;
                            return MoEOverlayRetirementFenceProgress::Pending;
                        }
                        if (local_state.readers ==
                            MoEOverlayRetirementReaderState::Active)
                        {
                            return MoEOverlayRetirementFenceProgress::Pending;
                        }
                        if (error)
                            error->clear();
                        return MoEOverlayRetirementFenceProgress::
                            ReadyToCloseAdmission;
                    }
                    if (local_state.readers ==
                        MoEOverlayRetirementReaderState::Active)
                    {
                        return MoEOverlayRetirementFenceProgress::Pending;
                    }
                    if (error)
                        error->clear();
                    return MoEOverlayRetirementFenceProgress::ReadyToRetire;
                }

            private:
                RecordingTransport *owner_ = nullptr;
            };

            explicit RecordingTransport(MoEOverlayResidencyAuthority *authority)
                : authority_(authority)
            {
            }

            MoEOverlayResidencyStageStart beginStage(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                calls.push_back("stage");
                staged_migrations = transaction.migrations;
                epoch_seen_during_stage = authority_->snapshot()->epoch;
                if (defer_start)
                {
                    return {
                        .status = MoEOverlayResidencyStageStartStatus::Deferred,
                        .error = "injected shadow-capacity backpressure",
                    };
                }
                return {
                    .status = MoEOverlayResidencyStageStartStatus::Started,
                    .wave = std::make_unique<Wave>(this),
                };
            }

            MoEOverlayResidencyAuthority *authority_ = nullptr;
            bool stage_ok = true;
            bool prepare_ok = true;
            bool publication_ok = true;
            bool defer_start = false;
            int stage_pending_polls = 0;
            int prepare_pending_polls = 0;
            int publication_pending_polls = 0;
            int retirement_admission_pending_polls = 0;
            int retirement_fence_polls = 0;
            uint64_t epoch_seen_during_stage = 0;
            uint64_t epoch_seen_during_prepare = 0;
            uint64_t epoch_seen_during_publication = 0;
            uint64_t epoch_seen_during_retire = 0;
            bool candidate_was_exact_addressable = false;
            std::vector<std::string> calls;
            std::vector<MoEOverlayTierMigration> staged_migrations;
        };

        const PerfStatRecord *findRecord(
            const std::vector<PerfStatRecord> &records,
            const std::string &name)
        {
            const auto found = std::find_if(
                records.begin(),
                records.end(),
                [&](const auto &record)
                {
                    return record.domain == "moe_overlay_residency" &&
                           record.name == name;
                });
            return found == records.end() ? nullptr : &*found;
        }

        class MoEOverlayResidencyOrderTest
            : public ::testing::TestWithParam<RoutedExpertOwnerOrder>
        {
        };
    } // namespace

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ThreeTierCapacityUsesClosedSlotFlowNotDirectionCountEquality)
    {
        const auto owner = [](int expert,
                              int tier,
                              int participant,
                              DeviceId device)
        {
            return MoEExpertOwner{
                .layer_idx = 0,
                .expert_id = expert,
                .tier_idx = tier,
                .owner_participant = participant,
                .device = device,
                .resident = true,
            };
        };
        const auto edge = [&](int expert,
                              int source_tier,
                              int source_participant,
                              DeviceId source_device,
                              int destination_tier,
                              int destination_participant,
                              DeviceId destination_device,
                              MoEOverlayTierMigrationDirection direction)
        {
            return MoEOverlayTierMigration{
                .layer_idx = 0,
                .expert_id = expert,
                .estimated_weight_bytes = 4096u,
                .direction = direction,
                .source = owner(
                    expert,
                    source_tier,
                    source_participant,
                    source_device),
                .destination = owner(
                    expert,
                    destination_tier,
                    destination_participant,
                    destination_device),
            };
        };

        /*
         * CPU-1 -> CUDA -> ROCm -> CPU-0 -> CPU-1 is one closed
         * participant cycle. Its thermal labels are intentionally asymmetric:
         * one direct promotion, two demotions, and one same-tier handoff.
         */
        const std::vector<MoEOverlayTierMigration> closed_cycle{
            edge(0, 2, 2, DeviceId::cpu(), 0, 0, DeviceId::cuda(0),
                 MoEOverlayTierMigrationDirection::Promotion),
            edge(1, 0, 0, DeviceId::cuda(0), 1, 1, DeviceId::rocm(0),
                 MoEOverlayTierMigrationDirection::Demotion),
            edge(2, 1, 1, DeviceId::rocm(0), 2, 3, DeviceId::cpu(),
                 MoEOverlayTierMigrationDirection::Demotion),
            edge(3, 2, 3, DeviceId::cpu(), 2, 2, DeviceId::cpu(),
                 MoEOverlayTierMigrationDirection::SamePriority),
        };

        const auto evidence =
            analyzeMoEOverlayMigrationCapacity(closed_cycle);
        EXPECT_TRUE(evidence.capacityPreserved());
        EXPECT_EQ(evidence.edges_checked, 4u);
        EXPECT_EQ(evidence.participant_flow_violations, 0u);
        EXPECT_EQ(evidence.tier_flow_violations, 0u);
        EXPECT_EQ(
            std::count_if(
                closed_cycle.begin(),
                closed_cycle.end(),
                [](const auto &migration)
                {
                    return migration.direction ==
                           MoEOverlayTierMigrationDirection::Promotion;
                }),
            1);
        EXPECT_EQ(
            std::count_if(
                closed_cycle.begin(),
                closed_cycle.end(),
                [](const auto &migration)
                {
                    return migration.direction ==
                           MoEOverlayTierMigrationDirection::Demotion;
                }),
            2);

        auto open_path = closed_cycle;
        open_path.pop_back();
        const auto broken_evidence =
            analyzeMoEOverlayMigrationCapacity(open_path);
        EXPECT_FALSE(broken_evidence.capacityPreserved());
        EXPECT_GT(broken_evidence.participant_flow_violations, 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        DeviceResidentDynamicAuthorityRejectsEveryHostPublicationPath)
    {
        auto plan = oneTierTwoParticipantPlan(
            RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        plan.authority_execution =
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = std::move(plan),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .perf_device = "one-tier-device-authority",
        });
        const uint64_t initial_epoch = authority.snapshot()->epoch;

        /*
         * The host object remains the immutable setup catalogue and lease
         * source. Once topology selects the captured participant authority,
         * however, none of its maintenance entry points may become a second
         * epoch writer—even if a future caller accidentally presents valid
         * histogram evidence or a transport implementation.
         */
        EXPECT_FALSE(authority.maintenanceWindowReady());
        EXPECT_THROW(
            (void)authority.progressHistogramWindow(),
            std::logic_error);
        EXPECT_THROW(
            (void)authority.freezeAndRotateHistogramWindow(),
            std::logic_error);
        EXPECT_THROW(
            (void)authority.proposeFromHistogram(),
            std::logic_error);
        EXPECT_THROW(
            (void)authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {100, 90, 80, 1, 1, 1})),
            std::logic_error);
        EXPECT_THROW(
            authority.installEconomyCertification(
                oneTierServiceProfile(),
                twoParticipantMigrationProfile(),
                MoEOverlayMigrationEconomyPolicy{}),
            std::logic_error);

        RecordingTransport transport(&authority);
        EXPECT_THROW(
            (void)authority.beginApply(
                MoEOverlayResidencyTransaction{}, transport),
            std::logic_error);
        EXPECT_THROW(
            (void)authority.advanceBackground(),
            std::logic_error);
        EXPECT_TRUE(transport.calls.empty());
        EXPECT_EQ(authority.snapshot()->epoch, initial_epoch);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        CpuAuthorityKeepsItsSoleLiveWriterOnTheHost)
    {
        auto plan = oneTierTwoParticipantPlan(
            RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        plan.domains.front().scope = ExecutionDomainScope::NODE_LOCAL;
        plan.domains.front().backend = CollectiveBackendType::UPI;
        plan.domains.front().participants = {
            GlobalDeviceAddress::cpu(0),
            GlobalDeviceAddress::cpu(1),
        };
        plan.domains.front().world_ranks = {0, 1};
        plan.authority_execution =
            MoEOverlayAuthorityExecutionKind::HostResident;
        ASSERT_EQ(
            resolveMoEOverlayAuthorityExecutionKind(plan),
            MoEOverlayAuthorityExecutionKind::
                HostResident);

        auto histogram = oneTierCpuHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = std::move(plan),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-two-cpu",
        });

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {100, 90, 80, 1, 1, 1}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 2u);
        for (const auto &migration : transaction.migrations)
        {
            EXPECT_TRUE(migration.source.device.is_cpu());
            EXPECT_TRUE(migration.destination.device.is_cpu());
            EXPECT_FALSE(migration.crossesTier());
        }

        RecordingTransport transport(&authority);
        ASSERT_EQ(
            authority.beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Published);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        OneTierAuthorityRebalancesParticipantSkewWithoutTierMovement)
    {
        ScopedPerfStats perf;
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-two-cuda",
        });

        const auto before = authority.snapshot();
        ASSERT_NE(before, nullptr);
        ASSERT_EQ(before->placement_plan->routed_tiers.size(), 1u);
        const auto evidence = frozenWindow(
            1, {100, 90, 80, 1, 1, 1});
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(evidence);

        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        ASSERT_EQ(transaction.migrations.size(), 2u)
            << "one capacity-preserving owner swap has two payload edges";
        ASSERT_EQ(
            transaction.previous->placement_plan->placements.size(),
            transaction.candidate->placement_plan->placements.size());
        for (std::size_t layer = 0;
             layer < transaction.previous->placement_plan->placements.size();
             ++layer)
        {
            EXPECT_EQ(
                transaction.previous->placement_plan->placements[layer]
                    .routed_expert_tier,
                transaction.candidate->placement_plan->placements[layer]
                    .routed_expert_tier)
                << "one-tier skew correction must not alter tier membership";
        }
        for (const auto &migration : transaction.migrations)
        {
            EXPECT_EQ(
                migration.direction,
                MoEOverlayTierMigrationDirection::SamePriority);
            EXPECT_FALSE(migration.crossesTier());
            EXPECT_FALSE(migration.crossesDomain());
            EXPECT_TRUE(migration.source.device.is_cuda());
            EXPECT_TRUE(migration.destination.device.is_cuda());
        }

        const auto participant_load = [&evidence](
                                          const MoELayeredExpertOwnership &owners)
        {
            std::array<uint64_t, 2> load{};
            for (int expert = 0; expert < 6; ++expert)
            {
                load[static_cast<std::size_t>(owners.owner(0, expert))] +=
                    evidence->activationCount(0, expert);
            }
            return load;
        };
        const auto before_load = participant_load(
            transaction.previous->layered_ownership);
        const auto after_load = participant_load(
            transaction.candidate->layered_ownership);
        EXPECT_LT(
            *std::max_element(after_load.begin(), after_load.end()) -
                *std::min_element(after_load.begin(), after_load.end()),
            *std::max_element(before_load.begin(), before_load.end()) -
                *std::min_element(before_load.begin(), before_load.end()));

        auto stats = authority.stats();
        EXPECT_EQ(stats.participant_rebalance_checks, 1u);
        EXPECT_EQ(stats.participant_rebalance_proposals, 1u);
        EXPECT_EQ(stats.participant_rebalance_owner_changes, 2u);

        RecordingTransport transport(&authority);
        const auto started = authority.beginApply(transaction, transport);
        ASSERT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        const auto published = authority.advanceBackground();
        ASSERT_EQ(
            published.status,
            MoEOverlayResidencyApplyStatus::Published);

        stats = authority.stats();
        EXPECT_EQ(stats.committed_migrations, 2u);
        EXPECT_EQ(stats.same_priority_moves, 2u);
        EXPECT_EQ(stats.promotions, 0u);
        EXPECT_EQ(stats.demotions, 0u);
        EXPECT_EQ(stats.cross_domain_migrations, 0u);

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        ASSERT_NE(
            findRecord(records, "participant_rebalance_checks"),
            nullptr);
        ASSERT_NE(
            findRecord(
                records,
                "participant_rebalance_owner_changes_proposed"),
            nullptr);
        ASSERT_NE(findRecord(records, "same_priority_moves"), nullptr);
        EXPECT_DOUBLE_EQ(
            findRecord(records, "same_priority_moves")->value,
            2.0);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ParticipantPlannerSearchesPastHottestColdestOvershoot)
    {
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-extrema-overshoot",
        });

        const auto evidence = frozenWindow(
            1, {10, 8, 0, 7, 5, 0});
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(evidence);

        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 2u);
        EXPECT_EQ(authority.stats().participant_rebalance_owner_changes, 2u);

        std::array<uint64_t, 2> resulting_load{};
        for (int expert = 0; expert < 6; ++expert)
        {
            resulting_load[static_cast<std::size_t>(
                transaction.candidate->layered_ownership.owner(
                    0, expert))] += evidence->activationCount(0, expert);
        }
        EXPECT_EQ(resulting_load[0], 15u);
        EXPECT_EQ(resulting_load[1], 15u)
            << "the planner must consider a non-coldest exchange when the "
               "hottest/coldest pair overshoots";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        StaticOneTierAuthorityRunsCheckWithoutParticipantMovement)
    {
        ScopedPerfStats perf;
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::StaticById),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Off,
            .histogram = nullptr,
            .perf_device = "one-tier-static",
        });
        RecordingTransport transport(&authority);

        const auto transaction = authority.proposeFromHistogram();
        ASSERT_TRUE(transaction.empty());
        const auto result = authority.beginApply(transaction, transport);
        EXPECT_EQ(
            result.status,
            MoEOverlayResidencyApplyStatus::StaticNoMovement);
        EXPECT_TRUE(transport.calls.empty());

        const auto stats = authority.stats();
        EXPECT_EQ(stats.static_no_movement_checks, 1u);
        EXPECT_EQ(stats.participant_rebalance_checks, 0u);
        EXPECT_EQ(stats.participant_rebalance_proposals, 0u);
        EXPECT_EQ(stats.same_priority_moves, 0u);
        EXPECT_EQ(stats.promotions, 0u);
        EXPECT_EQ(stats.demotions, 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        DynamicMaintenanceImprovesStaticByIdEpochWithoutRewritingIt)
    {
        auto initial = threeTierPlan(
            RoutedExpertResidencyPolicy::StaticById,
            RoutedExpertOwnerOrder::Ordinal);
        initial.placements.push_back({
            .layer = 0,
            .routed_expert_tier = {0, 0, 1, 1, 2, 2},
        });
        const auto declared_epoch_one =
            initial.placements.front().routed_expert_tier;
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});

        MoEOverlayResidencyAuthority authority({
            .initial_plan = std::move(initial),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .perf_device = "static-epoch-dynamic-maintenance",
        });

        const auto epoch_one = authority.snapshot();
        ASSERT_NE(epoch_one, nullptr);
        ASSERT_EQ(
            epoch_one->placement_plan->residency_policy,
            RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(
            epoch_one->placement_plan->placements.front()
                .routed_expert_tier,
            declared_epoch_one);
        EXPECT_EQ(authority.maintenanceMode(), MoERebalanceRuntimeMode::Dynamic);
        EXPECT_TRUE(authority.observesHistogram());
        EXPECT_TRUE(authority.migrationEnabled());
        ASSERT_EQ(epoch_one->owner_map.ownerFor(0, 4)->tier_name, "cold");

        const auto transaction = authority.proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_FALSE(transaction.empty());
        EXPECT_EQ(
            transaction.previous->placement_plan->residency_policy,
            RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(
            transaction.candidate->owner_map.ownerFor(0, 4)->tier_name,
            "hot")
            << "the hottest expert must be promoted from the adversarial epoch-one layout";
        EXPECT_GT(
            std::count_if(
                transaction.migrations.begin(),
                transaction.migrations.end(),
                [](const MoEOverlayTierMigration &migration)
                {
                    return migration.direction ==
                           MoEOverlayTierMigrationDirection::Promotion;
                }),
            0);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        CurrentBatchLLEPChildPinsExactDurableEpochUntilTransientRestore)
    {
        ScopedPerfStats perf;
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-llep",
        });

        std::array<uint32_t, 6> owners{};
        std::string error;
        auto llep_lease = authority.tryAcquireCurrentBatchLLEPLease(
            /*layer_idx=*/0,
            "cuda_domain",
            /*domain_participant_count=*/2,
            owners,
            &error);
        ASSERT_TRUE(llep_lease.has_value()) << error;
        EXPECT_EQ(
            llep_lease->purpose(),
            MoEOverlayResidencyAuthority::TicketLeasePurpose::
                CurrentBatchLLEP);
        EXPECT_EQ(llep_lease->epoch(), 1u);
        EXPECT_EQ(
            owners,
            (std::array<uint32_t, 6>{0, 0, 0, 1, 1, 1}));
        EXPECT_EQ(authority.activeTicketCount(), 1u);

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {100, 90, 80, 1, 1, 1}));
        ASSERT_FALSE(transaction.empty());
        RecordingTransport transport(&authority);
        ASSERT_EQ(
            authority.beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Published);
        EXPECT_EQ(authority.snapshot()->epoch, 2u);
        EXPECT_EQ(authority.pendingRetirementCount(), 1u)
            << "epoch one must remain alive while its LLEP child borrows sources";

        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(authority.pendingRetirementCount(), 1u);
        llep_lease.reset();
        EXPECT_EQ(authority.activeTicketCount(), 0u);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(authority.pendingRetirementCount(), 0u);

        const auto stats = authority.stats();
        EXPECT_EQ(stats.current_batch_llep_leases_acquired, 1u);
        EXPECT_EQ(stats.current_batch_llep_leases_released, 1u);
        EXPECT_EQ(stats.current_batch_llep_lease_rejections, 0u);
        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        ASSERT_NE(
            findRecord(records, "current_batch_llep_leases_acquired"),
            nullptr);
        ASSERT_NE(
            findRecord(records, "current_batch_llep_leases_released"),
            nullptr);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        CurrentBatchLLEPChildRejectsForeignDomainWithoutLeakingEpochLease)
    {
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::StaticById),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Off,
            .histogram = nullptr,
            .perf_device = "one-tier-llep-rejection",
        });
        std::array<uint32_t, 6> owners{};
        std::string error;

        const auto lease = authority.tryAcquireCurrentBatchLLEPLease(
            /*layer_idx=*/0,
            "foreign_domain",
            /*domain_participant_count=*/2,
            owners,
            &error);

        EXPECT_FALSE(lease.has_value());
        EXPECT_NE(error.find("participant count"), std::string::npos);
        EXPECT_EQ(authority.activeTicketCount(), 0u);
        EXPECT_EQ(
            authority.stats().current_batch_llep_lease_rejections,
            1u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        OneTierEconomyPricesReducedParticipantMakespan)
    {
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = oneTierServiceProfile(),
            .migration_cost_profile = twoParticipantMigrationProfile(),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 2048,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-economy",
        });

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {100, 90, 80, 1, 1, 1}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_FALSE(transaction.empty());
        EXPECT_TRUE(transaction.economy.enabled);
        EXPECT_GT(transaction.economy.projected_service_gain_ns, 0u)
            << "same-tier movement saves parallel participant makespan even "
               "though aggregate tier work is unchanged";
        EXPECT_GT(transaction.economy.projected_net_benefit_ns, 0u);
        EXPECT_EQ(transaction.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(transaction.migrations.size(), 2u);
        EXPECT_EQ(transaction.economy.projected_transfer_and_repack_ns, 1u)
            << "A parallel swap must charge one measured critical path";
        EXPECT_EQ(
            transaction.economy.projected_inference_interference_ns,
            1u)
            << "Paired overlap interference must not be charged per edge";
        EXPECT_TRUE(std::all_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return migration.direction ==
                       MoEOverlayTierMigrationDirection::SamePriority;
            }));
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ParticipantSpecificCostRejectsRawCountBalanceOntoSlowerPeer)
    {
        auto histogram = oneTierHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = oneTierTwoParticipantPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = asymmetricOneTierServiceProfile(),
            .migration_cost_profile = twoParticipantMigrationProfile(),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 2048,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1300,
                .minimum_improvement_per_mille = 50,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "one-tier-asymmetric-economy",
        });

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {100, 90, 80, 1, 1, 1}));
        ASSERT_TRUE(transaction.valid());
        EXPECT_TRUE(transaction.empty())
            << "raw activation balance must not move a hot expert onto a "
               "participant whose measured service makes the critical path worse";
        EXPECT_GT(transaction.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(transaction.economy.projected_service_gain_ns, 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        PromotionThatLeavesAnotherParticipantCriticalIsRejected)
    {
        auto histogram = histogramWithCounts(std::vector<uint64_t>(8, 0u));
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierNodeLocalPlan(
                RoutedExpertOwnerOrder::Ordinal,
                /*accelerator_capacity=*/1),
            .model_metadata = eightExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(),
            .migration_cost_profile =
                threeParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 2048,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {.enabled = false},
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "tier-promotion-critical-path",
        });

        std::vector<uint64_t> counts(8, 0u);
        int promoted_candidate = -1;
        std::vector<int> retained_bottleneck_experts;
        for (const auto &owner : authority.snapshot()->owner_map.owners())
        {
            if (owner.tier_idx != 1)
                continue;
            if (owner.owner_participant == 1 && promoted_candidate < 0)
                promoted_candidate = owner.expert_id;
            if (owner.owner_participant == 2)
                retained_bottleneck_experts.push_back(owner.expert_id);
        }
        ASSERT_GE(promoted_candidate, 0);
        ASSERT_GE(retained_bottleneck_experts.size(), 2u);
        counts[static_cast<std::size_t>(promoted_candidate)] = 100u;
        counts[static_cast<std::size_t>(retained_bottleneck_experts[0])] = 60u;
        counts[static_cast<std::size_t>(retained_bottleneck_experts[1])] = 60u;

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, counts));
        ASSERT_TRUE(transaction.valid());
        EXPECT_TRUE(transaction.empty())
            << "accelerating one expert is not a payoff while another remote "
               "participant still gates the layer";
        EXPECT_GT(transaction.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(transaction.economy.projected_service_gain_ns, 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        DecodeGainCannotCrossSubsidizeAPrefillRegression)
    {
        auto histogram = histogramWithCounts(std::vector<uint64_t>(8, 0u));
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierNodeLocalPlan(
                RoutedExpertOwnerOrder::Ordinal,
                /*accelerator_capacity=*/1),
            .model_metadata = eightExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(),
            .migration_cost_profile =
                threeParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 2048,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {.enabled = false},
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "phase-economy-cross-subsidy",
        });

        std::vector<uint64_t> decode(8, 0u);
        std::vector<uint64_t> prefill(8, 0u);
        const auto snapshot = authority.snapshot();
        const auto accelerator = std::find_if(
            snapshot->owner_map.owners().begin(),
            snapshot->owner_map.owners().end(),
            [](const auto &owner) { return owner.tier_idx == 0; });
        const auto cpu = std::find_if(
            snapshot->owner_map.owners().begin(),
            snapshot->owner_map.owners().end(),
            [](const auto &owner) { return owner.tier_idx == 1; });
        ASSERT_NE(accelerator, snapshot->owner_map.owners().end());
        ASSERT_NE(cpu, snapshot->owner_map.owners().end());

        decode[static_cast<std::size_t>(cpu->expert_id)] = 1000u;
        decode[static_cast<std::size_t>(accelerator->expert_id)] = 10u;
        prefill[static_cast<std::size_t>(cpu->expert_id)] = 10u;
        prefill[static_cast<std::size_t>(accelerator->expert_id)] = 200u;
        const std::vector<uint64_t> grouped(8, 0u);

        const auto rejected = authority.proposeFromFrozenHistogramWindow(
            phasedFrozenWindow(1, {decode, prefill, grouped}));
        ASSERT_TRUE(rejected.valid());
        EXPECT_TRUE(rejected.empty())
            << "a decode win must never authorize a prefill regression";
        EXPECT_GT(rejected.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(rejected.economy.projected_service_gain_ns, 0u);
    }

    TEST_P(
        MoEOverlayResidencyOrderTest,
        ThreeTierWavePromotesHottestDemotesColdAndPublishesAfterCommit)
    {
        ScopedPerfStats perf;
        /*
         * Initial by-id residency is hot={0,1}, warm={2,3}, cold={4,5}.
         * The evidence produces hot={4,0}, warm={2,5}, cold={1,3}, forcing
         * traffic into and out of all three physical domains in one wave.
         */
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                GetParam()),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .perf_device = "CUDA:0",
        });
        RecordingTransport transport(&authority);

        const auto before = authority.snapshot();
        ASSERT_EQ(before->epoch, 1u);
        ASSERT_EQ(before->owner_map.ownerFor(0, 0)->tier_name, "hot");
        ASSERT_EQ(before->owner_map.ownerFor(0, 4)->tier_name, "cold");

        const auto transaction = authority.proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 4u);
        ASSERT_EQ(transaction.migration_cycles.size(), 2u);
        for (const auto &cycle : transaction.migration_cycles)
        {
            EXPECT_TRUE(cycle.valid(transaction.migrations));
            EXPECT_EQ(cycle.migration_indices.size(), 2u);
        }
        ASSERT_EQ(transaction.shadow_requirements.size(), 3u);
        const auto requiredSlotsForTier = [&](int tier_idx)
        {
            size_t slots = 0;
            for (const auto &requirement : transaction.shadow_requirements)
            {
                if (requirement.tier_idx == tier_idx)
                    slots += requirement.slot_count;
            }
            return slots;
        };
        EXPECT_EQ(requiredSlotsForTier(0), 1u);
        EXPECT_EQ(requiredSlotsForTier(1), 1u);
        EXPECT_EQ(requiredSlotsForTier(2), 2u);
        EXPECT_EQ(authority.snapshot()->epoch, 1u);

        const auto started = authority.beginApply(transaction, transport);
        ASSERT_TRUE(started.ok()) << started.error;
        EXPECT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(started.published_epoch, 1u);
        EXPECT_EQ(authority.snapshot()->epoch, 1u);

        const auto preparing = authority.advanceBackground();
        ASSERT_TRUE(preparing.ok()) << preparing.error;
        EXPECT_EQ(
            preparing.status,
            MoEOverlayResidencyApplyStatus::Preparing);
        EXPECT_EQ(authority.snapshot()->epoch, 1u)
            << "The candidate must remain private until preparation readiness";

        const auto publishing = authority.advanceBackground();
        ASSERT_TRUE(publishing.ok()) << publishing.error;
        EXPECT_EQ(
            publishing.status,
            MoEOverlayResidencyApplyStatus::Publishing);
        EXPECT_EQ(authority.snapshot()->epoch, 1u)
            << "Ordinary admission must remain on E while selectors fan out";
        EXPECT_TRUE(transport.candidate_was_exact_addressable)
            << "Device-selected E+1 tickets require an exact host snapshot";

        const auto result = authority.advanceBackground();
        ASSERT_TRUE(result.ok()) << result.error;
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::Published);
        EXPECT_EQ(result.published_epoch, 2u);
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{
                      "stage",
                      "prepare",
                      "publish",
                      "authority-published",
                      "retire"}));
        EXPECT_EQ(transport.epoch_seen_during_stage, 1u);
        EXPECT_EQ(transport.epoch_seen_during_prepare, 1u);
        EXPECT_EQ(transport.epoch_seen_during_publication, 1u);
        EXPECT_EQ(transport.epoch_seen_during_retire, 2u)
            << "Old residency may retire only after atomic owner publication";

        const auto after = authority.snapshot();
        ASSERT_EQ(after->owner_map.ownerFor(0, 4)->tier_name, "hot");
        ASSERT_EQ(after->owner_map.ownerFor(0, 0)->tier_name, "hot");
        ASSERT_EQ(after->owner_map.ownerFor(0, 2)->tier_name, "warm");
        ASSERT_EQ(after->owner_map.ownerFor(0, 5)->tier_name, "warm");
        ASSERT_EQ(after->owner_map.ownerFor(0, 1)->tier_name, "cold");
        ASSERT_EQ(after->owner_map.ownerFor(0, 3)->tier_name, "cold");

        const auto stats = authority.stats();
        EXPECT_EQ(stats.committed_waves, 1u);
        EXPECT_EQ(stats.committed_migrations, 4u);
        EXPECT_EQ(stats.committed_cycles, 2u);
        EXPECT_EQ(stats.promotions, 2u);
        EXPECT_EQ(stats.demotions, 2u);
        EXPECT_EQ(stats.cross_domain_migrations, 4u);
        EXPECT_EQ(stats.cross_rank_migrations, 4u);
        EXPECT_EQ(stats.cross_backend_migrations, 4u);
        EXPECT_EQ(stats.background_waves_started, 1u);
        EXPECT_EQ(stats.old_epoch_retirements, 1u);

        const auto movement_ledger = authority.movementLedger();
        ASSERT_TRUE(movement_ledger.complete());
        ASSERT_EQ(movement_ledger.edges.size(), 4u);
        EXPECT_EQ(
            std::count_if(
                movement_ledger.edges.begin(),
                movement_ledger.edges.end(),
                [](const auto &edge)
                {
                    return edge.direction ==
                           MoEOptimizationMovementDirection::Promotion;
                }),
            2);
        EXPECT_EQ(
            std::count_if(
                movement_ledger.edges.begin(),
                movement_ledger.edges.end(),
                [](const auto &edge)
                {
                    return edge.direction ==
                           MoEOptimizationMovementDirection::Demotion;
                }),
            2);
        for (const auto &edge : movement_ledger.edges)
        {
            EXPECT_TRUE(edge.valid());
            EXPECT_EQ(edge.authority, MoEOptimizationAuthority::Host);
            EXPECT_EQ(edge.transaction, 2u);
            EXPECT_EQ(edge.candidate_epoch, 2u);
            EXPECT_EQ(edge.cycle_size, 2u);
            EXPECT_GT(edge.activation_count, 0u);
        }

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        ASSERT_NE(findRecord(records, "committed_expert_migrations"), nullptr);
        EXPECT_DOUBLE_EQ(
            findRecord(records, "committed_expert_migrations")->value,
            4.0);
        ASSERT_NE(findRecord(records, "promotions"), nullptr);
        EXPECT_DOUBLE_EQ(findRecord(records, "promotions")->value, 2.0);
        ASSERT_NE(findRecord(records, "demotions"), nullptr);
        EXPECT_DOUBLE_EQ(findRecord(records, "demotions")->value, 2.0);
        ASSERT_NE(findRecord(records, "cross_domain_migrations"), nullptr);
        EXPECT_DOUBLE_EQ(
            findRecord(records, "cross_domain_migrations")->value,
            4.0);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        AsyncHistogramDrainRemainsReadyWhileInferenceContinues)
    {
        auto histogram = histogramWithCounts({0, 0, 0, 0, 0, 0});
        for (int token = 0; token < 4; ++token)
            histogram->recordTokenBoundary(0);

        int drain_polls = 0;
        histogram->registerRuntimeHistogramDrain(
            [&]()
            {
                ++drain_polls;
                if (drain_polls == 1)
                    return RuntimeExpertHistogramDrainResult::pending();
                const uint64_t device_counts[6] = {0, 0, 0, 0, 9, 0};
                histogram->mergeLayerCounts(
                    0,
                    device_counts,
                    6,
                    /*count_window_tokens=*/false,
                    ExpertHistogramSource::GroupedVerifier);
                return RuntimeExpertHistogramDrainResult::ready();
            });

        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .histogram_max_window_tokens = 16u,
            .histogram_window_growth_factor = 4.0,
            .perf_device = "CUDA:0",
        });

        ASSERT_TRUE(authority.maintenanceWindowReady());
        const auto pending = authority.progressHistogramWindow();
        EXPECT_EQ(
            pending.progress,
            MoEOverlayHistogramWindowProgress::Pending);
        EXPECT_FALSE(pending.window);
        EXPECT_TRUE(authority.maintenanceWindowReady())
            << "An in-flight drain remains schedulable without another token";

        /* Simulate inference admission while the device event is pending. The
         * RCU host writer does not wait for maintenance and must land in the
         * exact generation being prepared. */
        const int experts[2] = {1, 2};
        const float weights[2] = {0.75F, 0.25F};
        histogram->record(0, experts, weights, 2);

        const auto ready = authority.progressHistogramWindow();
        ASSERT_EQ(
            ready.progress,
            MoEOverlayHistogramWindowProgress::Ready);
        ASSERT_TRUE(ready.window);
        EXPECT_EQ(drain_polls, 2);
        EXPECT_EQ(ready.window->token_count, 5u);
        EXPECT_EQ(ready.window->activationCount(0, 1), 1u);
        EXPECT_EQ(ready.window->activationCount(0, 2), 1u);
        EXPECT_EQ(
            ready.window->activationCount(
                ExpertHistogramSource::GroupedVerifier,
                0,
                4),
            9u);
        EXPECT_EQ(histogram->windowSize(), 16);
        EXPECT_EQ(
            authority.optimizationDemandWindow().capacity_routed_rows,
            16u);
        EXPECT_FALSE(authority.maintenanceWindowReady());
    }

    INSTANTIATE_TEST_SUITE_P(
        OrdinalAndRandom,
        MoEOverlayResidencyOrderTest,
        ::testing::Values(
            RoutedExpertOwnerOrder::Ordinal,
            RoutedExpertOwnerOrder::Random));

    TEST(
        Test__MoEOverlayResidencyAuthority,
        OneSlotBomAdmitsHighestBenefitClosedCycleForBothOwnerOrders)
    {
        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            SCOPED_TRACE(routedExpertOwnerOrderToString(owner_order));
            auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
            MoEOverlayResidencyAuthority authority({
                .initial_plan = threeTierPlan(
                    RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                    owner_order),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
                .shadow_slots_per_endpoint_layer = 1,
                .max_concurrent_cycles = 1,
                .perf_device = "bounded-three-tier",
            });

            const auto transaction = authority.proposeFromHistogram();
            ASSERT_TRUE(transaction.valid());
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);
            ASSERT_EQ(transaction.migrations.size(), 2u);
            for (const auto &requirement : transaction.shadow_requirements)
                EXPECT_EQ(requirement.slot_count, 1u);

            /*
             * e4->hot / e1->cold has thermal score 160; e5->warm /
             * e3->cold scores 60. The bounded wave must choose the former.
             */
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 4)->tier_name,
                "hot");
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 1)->tier_name,
                "cold");
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 5)->tier_name,
                "cold");
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 3)->tier_name,
                "warm");

            const auto stats = authority.stats();
            EXPECT_EQ(stats.capacity_bounded_proposals, 1u);
            EXPECT_EQ(stats.bounded_candidate_snapshot_builds, 1u)
                << "A saturated one-cycle wave must not rebuild omitted candidates";
            EXPECT_EQ(stats.target_migrations_omitted, 2u);
            EXPECT_EQ(stats.target_cycles_omitted, 1u);
        }
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        PolicyRejectedCycleDoesNotMasqueradeAsCapacityPressure)
    {
        ScopedPerfStats perf;
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = threeTierServiceProfile(),
            .migration_cost_profile =
                threeParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    /*
                     * The hot/cold exchange saves 8,000 ns in this measured
                     * window while the warm/cold exchange saves 6,000 ns.
                     * Admit exactly the former without exhausting either of
                     * the two available transfer slots.
                     */
                    .minimum_net_benefit_ns = 7'000,
                    .minimum_residency_generations = 0,
                },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 2,
            .perf_device = "policy-not-capacity",
        });

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        ASSERT_TRUE(transaction.host_admission.has_value());
        const auto &typed_admission = *transaction.host_admission;
        EXPECT_TRUE(typed_admission.valid());
        EXPECT_EQ(typed_admission.candidate_cycles, 2u);
        EXPECT_EQ(typed_admission.policy_eligible_cycles, 1u);
        EXPECT_EQ(typed_admission.admitted_candidate_cycles, 1u);
        EXPECT_EQ(typed_admission.admitted_physical_cycles, 1u);
        EXPECT_EQ(
            typed_admission.individual_policy_rejected_cycles,
            1u);
        EXPECT_TRUE(typed_admission.policy_bounded);
        EXPECT_FALSE(typed_admission.capacity_bounded);
        EXPECT_GT(transaction.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(authority.stats().capacity_bounded_proposals, 0u);
        EXPECT_EQ(authority.stats().target_cycles_omitted, 0u);
        EXPECT_EQ(authority.stats().target_migrations_omitted, 0u);

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto *const admission =
            findRecord(records, "cycle_axis_admission");
        ASSERT_NE(admission, nullptr);
        EXPECT_EQ(admission->tags.at("candidate_cycles"), "2");
        EXPECT_EQ(admission->tags.at("policy_eligible_cycles"), "1");
        EXPECT_EQ(admission->tags.at("admitted_cycles"), "1");
        EXPECT_EQ(
            admission->tags.at("admitted_candidate_cycles"), "1");
        EXPECT_EQ(
            admission->tags.at("physical_cycle_recomposition"), "false");
        EXPECT_EQ(
            admission->tags.at("individual_policy_rejected_cycles"),
            "1");
        EXPECT_EQ(
            admission->tags.at("dependent_payoff_rejected_cycles"),
            "0");
        EXPECT_EQ(admission->tags.at("capacity_rejected_cycles"), "0");
        EXPECT_EQ(admission->tags.at("capacity_bounded"), "false");
        EXPECT_EQ(admission->tags.at("policy_bounded"), "true");
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        DelayedEconomyCertificationIsOneShotAndPrecedesFirstProposal)
    {
        auto histogram = histogramWithCounts({0, 0, 0, 0, 0, 0});
        auto service_profile = threeTierServiceProfile();
        auto migration_profile = threeParticipantMigrationProfile(
            /*transfer_and_repack_ns=*/1,
            /*inference_interference_ns=*/1);
        const MoEOverlayMigrationEconomyPolicy policy{
            .historical_window_weight = 0,
            .current_window_weight = 1,
            .payoff_horizon_tokens = 370,
            .minimum_net_benefit_ns = 0,
            .minimum_residency_generations = 0,
        };

        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        EXPECT_FALSE(authority.hasEconomyCertification());

        /* Graph setup may inspect the immutable initial epoch before profiling. */
        auto setup_ticket = authority.tryAcquireTicketSnapshot();
        ASSERT_TRUE(setup_ticket.has_value());
        setup_ticket.reset();

        ASSERT_EQ(
            authority.progressEconomyEvidenceRebase(),
            MoEOverlayHistogramRebaseProgress::Complete);
        authority.installEconomyCertification(
            service_profile,
            migration_profile,
            policy);
        EXPECT_TRUE(authority.hasEconomyCertification());
        EXPECT_FALSE(authority.optimizationDemandActive());
        EXPECT_THROW(
            (void)authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {90, 20, 80, 10, 100, 70})),
            std::logic_error);
        EXPECT_EQ(
            authority.activateOptimizationDemandAtRequestBoundary(),
            MoEOverlayDemandActivationResult::Activated);
        EXPECT_TRUE(authority.optimizationDemandActive());

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        EXPECT_TRUE(transaction.economy.enabled);
        EXPECT_EQ(
            transaction.economy.service_profile_identity,
            service_profile->identity);
        EXPECT_EQ(
            transaction.economy.migration_profile_identity,
            migration_profile->identity);
        ASSERT_EQ(transaction.migration_cycles.size(), 2u);
        EXPECT_EQ(transaction.economy.projected_transfer_and_repack_ns, 2u)
            << "Separately calibrated cycles remain conservatively additive";
        EXPECT_EQ(
            transaction.economy.projected_inference_interference_ns,
            2u);

        EXPECT_THROW(
            authority.installEconomyCertification(
                service_profile,
                migration_profile,
                policy),
            std::logic_error);

        MoEOverlayResidencyAuthority already_proposed({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        (void)already_proposed.proposeFromFrozenHistogramWindow(
            frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        EXPECT_THROW(
            already_proposed.installEconomyCertification(
                service_profile,
                migration_profile,
                policy),
            std::logic_error);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        EconomyActivationAsynchronouslyDiscardsCalibrationDemand)
    {
        auto histogram = histogramWithCounts({7, 6, 5, 4, 3, 2});
        histogram->recordTokenBoundary(0);
        const auto calibration_generation = histogram->windowGeneration();

        int drain_polls = 0;
        histogram->registerRuntimeHistogramDrain(
            [&]()
            {
                ++drain_polls;
                return drain_polls == 1
                           ? RuntimeExpertHistogramDrainResult::pending()
                           : RuntimeExpertHistogramDrainResult::ready();
            });

        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });

        EXPECT_FALSE(authority.maintenanceWindowReady())
            << "Certification rebase must not require a full policy window";
        EXPECT_EQ(
            authority.progressEconomyEvidenceRebase(),
            MoEOverlayHistogramRebaseProgress::Pending);
        EXPECT_FALSE(authority.maintenanceWindowReady())
            << "A certification drain must never masquerade as proposal work";
        EXPECT_THROW(
            (void)authority.progressHistogramWindow(),
            std::logic_error)
            << "The shared drain lane has one typed purpose per generation";

        /* Inference remains non-blocking while device evidence drains. Traffic
         * after quarantine begins is deliberately excluded rather than being
         * allowed to leak into the new movement generation. */
        const int experts[2] = {0, 1};
        const float weights[2] = {0.75F, 0.25F};
        histogram->record(0, experts, weights, 2);
        histogram->recordTokenBoundary(0);

        EXPECT_EQ(
            authority.progressEconomyEvidenceRebase(),
            MoEOverlayHistogramRebaseProgress::Complete);
        EXPECT_EQ(drain_polls, 2);
        EXPECT_EQ(
            histogram->windowGeneration(),
            calibration_generation + 1);
        for (int expert = 0; expert < 6; ++expert)
            EXPECT_EQ(histogram->activationCount(0, expert), 0u);
        EXPECT_FALSE(authority.maintenanceWindowReady());
        EXPECT_EQ(
            authority.progressEconomyEvidenceRebase(),
            MoEOverlayHistogramRebaseProgress::Complete)
            << "A completed typed edge is idempotent until installation";

        authority.installEconomyCertification(
            threeTierServiceProfile(),
            threeParticipantMigrationProfile(1, 1),
            {
                .historical_window_weight = 0,
                .current_window_weight = 1,
                .payoff_horizon_tokens = 8,
                .minimum_residency_generations = 0,
            });
        EXPECT_TRUE(authority.hasEconomyCertification());
        EXPECT_FALSE(authority.optimizationDemandActive());

        histogram->record(0, experts, weights, 2);
        histogram->recordTokenBoundary(0);
        EXPECT_EQ(histogram->windowTokenCount(), 0u)
            << "The tail of the certification request remains quarantined";

        EXPECT_EQ(
            authority.activateOptimizationDemandAtRequestBoundary(),
            MoEOverlayDemandActivationResult::Activated);
        EXPECT_TRUE(authority.optimizationDemandActive());
        histogram->record(0, experts, weights, 2);
        EXPECT_EQ(histogram->activationCount(0, 0), 1u);
        EXPECT_EQ(histogram->activationCount(0, 1), 1u);
        EXPECT_THROW(
            (void)authority.progressEconomyEvidenceRebase(),
            std::logic_error);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        MeasuredPayoffGateRejectsUneconomicalCyclesWithoutStartingTransport)
    {
        ScopedPerfStats perf;
        auto histogram = histogramWithCounts({0, 0, 0, 0, 0, 0});
        auto service_profile = threeTierServiceProfile();
        auto migration_profile = threeParticipantMigrationProfile(
            /*transfer_and_repack_ns=*/1'000'000'000,
            /*inference_interference_ns=*/10'000);
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = service_profile,
            .migration_cost_profile = migration_profile,
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 1,
                    .minimum_residency_generations = 0,
                },
            .perf_device = "priority-tiers",
        });
        RecordingTransport transport(&authority);

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        ASSERT_TRUE(transaction.valid());
        EXPECT_TRUE(transaction.empty());
        EXPECT_TRUE(transaction.economy.enabled);
        EXPECT_EQ(transaction.economy.projected_service_gain_ns, 0u);
        EXPECT_EQ(transaction.economy.projected_net_benefit_ns, 0u);
        EXPECT_GT(transaction.economy.payoff_rejected_cycles, 0u);
        EXPECT_EQ(transaction.economy.residency_rejected_cycles, 0u);
        EXPECT_EQ(
            transaction.candidate->placement_plan->placements.front()
                .routed_expert_tier,
            transaction.previous->placement_plan->placements.front()
                .routed_expert_tier);

        const auto result = authority.beginApply(transaction, transport);
        EXPECT_EQ(
            result.status,
            MoEOverlayResidencyApplyStatus::DynamicNoMovement);
        EXPECT_TRUE(transport.calls.empty());
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_EQ(authority.stats().economy_proposals, 1u);
        EXPECT_GT(authority.stats().payoff_rejected_cycles, 0u);

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        ASSERT_NE(findRecord(records, "payoff_rejected_cycles"), nullptr);
        EXPECT_GT(findRecord(records, "payoff_rejected_cycles")->value, 0.0);
        ASSERT_NE(findRecord(records, "projected_net_benefit_ns"), nullptr);
        EXPECT_DOUBLE_EQ(
            findRecord(records, "projected_net_benefit_ns")->value,
            0.0);
        const auto *rejected_gain = findRecord(
            records,
            "closest_rejected_projected_service_gain_ns");
        const auto *rejected_transfer = findRecord(
            records,
            "closest_rejected_transfer_and_repack_ns");
        const auto *rejected_interference = findRecord(
            records,
            "closest_rejected_inference_interference_ns");
        const auto *rejected_shortfall = findRecord(
            records,
            "closest_rejected_payoff_shortfall_ns");
        ASSERT_NE(rejected_gain, nullptr);
        ASSERT_NE(rejected_transfer, nullptr);
        ASSERT_NE(rejected_interference, nullptr);
        ASSERT_NE(rejected_shortfall, nullptr);
        EXPECT_GT(rejected_gain->value, 0.0);
        EXPECT_EQ(rejected_transfer->value, 1'000'000'000.0);
        EXPECT_EQ(rejected_interference->value, 10'000.0);
        EXPECT_GT(rejected_shortfall->value, 0.0);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        MinimumResidencyAgeStartsOnlyAfterCommitAndPreventsImmediateReversal)
    {
        auto histogram = histogramWithCounts({0, 0, 0, 0, 0, 0});
        auto service_profile = threeTierServiceProfile();
        auto migration_profile = threeParticipantMigrationProfile(
            /*transfer_and_repack_ns=*/1,
            /*inference_interference_ns=*/1);
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = service_profile,
            .migration_cost_profile = migration_profile,
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 2,
                },
        });
        RecordingTransport transport(&authority);

        const auto first = authority.proposeFromFrozenHistogramWindow(
            frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        ASSERT_FALSE(first.empty());
        ASSERT_GT(first.economy.projected_net_benefit_ns, 0u);
        ASSERT_EQ(
            authority.beginApply(first, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Published);
        ASSERT_EQ(authority.snapshot()->epoch, 2u);

        const auto immediate_reverse =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(2, {100, 90, 80, 70, 20, 10}));
        ASSERT_TRUE(immediate_reverse.valid());
        EXPECT_TRUE(immediate_reverse.empty());
        EXPECT_GT(
            immediate_reverse.economy.residency_rejected_cycles,
            0u);
        EXPECT_EQ(authority.snapshot()->epoch, 2u);

        const auto aged_reverse =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(3, {100, 90, 80, 70, 20, 10}));
        ASSERT_TRUE(aged_reverse.valid());
        EXPECT_FALSE(aged_reverse.empty());
        EXPECT_EQ(aged_reverse.economy.residency_rejected_cycles, 0u);
        EXPECT_GT(aged_reverse.economy.projected_net_benefit_ns, 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        IntegerSmoothingSuppressesOneWindowReversalThenConverges)
    {
        auto histogram = histogramWithCounts({0, 0, 0, 0, 0, 0});
        auto service_profile = threeTierServiceProfile();
        auto migration_profile = threeParticipantMigrationProfile(
            /*transfer_and_repack_ns=*/1,
            /*inference_interference_ns=*/0);
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = service_profile,
            .migration_cost_profile = migration_profile,
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 3,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
        });
        RecordingTransport transport(&authority);

        const auto first = authority.proposeFromFrozenHistogramWindow(
            frozenWindow(1, {90, 20, 80, 10, 100, 70}));
        ASSERT_FALSE(first.empty());
        ASSERT_EQ(
            authority.beginApply(first, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Published);

        const std::vector<uint64_t> reversed{
            20, 100, 10, 90, 70, 80};
        const auto one_noisy_window =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(2, reversed));
        ASSERT_TRUE(one_noisy_window.valid());
        EXPECT_TRUE(one_noisy_window.empty())
            << "A single equal-scale reversal must not churn residency";

        const auto sustained_reversal =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(3, reversed));
        ASSERT_TRUE(sustained_reversal.valid());
        EXPECT_FALSE(sustained_reversal.empty())
            << "Repeated evidence must eventually overcome smoothing";
        ASSERT_NE(sustained_reversal.histogram_window, nullptr);
        bool retained_smoothed_value = false;
        for (const auto &migration : sustained_reversal.migrations)
        {
            const uint64_t planning_count =
                sustained_reversal.histogram_window->activationCount(
                    migration.layer_idx, migration.expert_id);
            EXPECT_EQ(migration.activation_count, planning_count)
                << "every executable migration must retain its exact planning evidence";
            const std::size_t raw_index =
                static_cast<std::size_t>(migration.layer_idx) *
                    static_cast<std::size_t>(
                        sustained_reversal.histogram_window->num_experts) +
                static_cast<std::size_t>(migration.expert_id);
            retained_smoothed_value |= planning_count != reversed[raw_index];
        }
        EXPECT_TRUE(retained_smoothed_value)
            << "generation-three movement must publish smoothed evidence, not the raw input bank";
        EXPECT_GT(
            sustained_reversal.economy.projected_net_benefit_ns,
            0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ArbitraryFourTierRotationFormsOneClosedShadowSafeCycle)
    {
        auto histogram = fourTierRotationHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = fourTierCyclePlan(),
            .model_metadata = fourTierMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });

        const auto transaction = authority.proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 4u);
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        const auto &cycle = transaction.migration_cycles.front();
        ASSERT_TRUE(cycle.valid(transaction.migrations));
        EXPECT_EQ(cycle.migration_indices.size(), 4u);

        std::vector<size_t> required_by_tier(4, 0);
        for (const auto &requirement : transaction.shadow_requirements)
        {
            ASSERT_GE(requirement.tier_idx, 0);
            ASSERT_LT(requirement.tier_idx, 4);
            required_by_tier[static_cast<size_t>(requirement.tier_idx)] +=
                requirement.slot_count;
        }
        EXPECT_EQ(required_by_tier, (std::vector<size_t>{1, 1, 1, 1}))
            << "A simple N-tier cycle needs one inactive destination per tier";

        int current_tier =
            transaction.migrations[cycle.migration_indices.front()]
                .source.tier_idx;
        for (const size_t edge_index : cycle.migration_indices)
        {
            const auto &edge = transaction.migrations[edge_index];
            EXPECT_EQ(edge.source.tier_idx, current_tier);
            current_tier = edge.destination.tier_idx;
        }
        EXPECT_EQ(
            current_tier,
            transaction.migrations[cycle.migration_indices.front()]
                .source.tier_idx);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        NodeLocalTierRetainsOwnersAndAvoidsGratuitousSameTierMovement)
    {
        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            SCOPED_TRACE(routedExpertOwnerOrderToString(owner_order));
            auto histogram = nodeLocalRotationHistogram();
            MoEOverlayResidencyAuthority authority({
                .initial_plan = nodeLocalThreeTierPlan(owner_order),
                .model_metadata = fourTierMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .shadow_slots_per_endpoint_layer = 1,
                .max_concurrent_cycles = 1,
                .perf_device =
                    "cuda-priority0/rocm-priority1/cpu-nodelocal-priority2",
            });

            const auto before = authority.snapshot();
            const int retained_cpu_owner =
                before->owner_map.ownerFor(0, 2)->owner_participant;
            const auto transaction = authority.proposeFromHistogram();
            ASSERT_TRUE(transaction.valid());
            ASSERT_EQ(transaction.migrations.size(), 3u);
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);
            const auto &cycle = transaction.migration_cycles.front();
            ASSERT_TRUE(cycle.valid(transaction.migrations));
            ASSERT_EQ(cycle.migration_indices.size(), 3u);

            const auto same_tier_moves = std::count_if(
                transaction.migrations.begin(),
                transaction.migrations.end(),
                [](const auto &migration)
                {
                    return migration.source.tier_idx ==
                           migration.destination.tier_idx;
                });
            EXPECT_EQ(same_tier_moves, 0)
                << "A retained expert must not move between CPU participants";

            int current_participant =
                transaction.migrations[cycle.migration_indices.front()]
                    .source.owner_participant;
            const int first_participant = current_participant;
            for (const std::size_t migration_index : cycle.migration_indices)
            {
                const auto &migration =
                    transaction.migrations[migration_index];
                EXPECT_EQ(
                    migration.source.owner_participant,
                    current_participant);
                current_participant =
                    migration.destination.owner_participant;
            }
            EXPECT_EQ(
                current_participant,
                first_participant);

            for (const auto &requirement : transaction.shadow_requirements)
                EXPECT_EQ(requirement.slot_count, 1u);
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 3)->tier_name,
                "hot");
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 0)->tier_name,
                "warm");
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 2)
                    ->owner_participant,
                retained_cpu_owner);
        }
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        BoundedEconomyCycleKeepsIdentityAcrossNodeLocalParticipants)
    {
        for (const auto owner_order : {
                 RoutedExpertOwnerOrder::Ordinal,
                 RoutedExpertOwnerOrder::Random})
        {
            SCOPED_TRACE(routedExpertOwnerOrderToString(owner_order));
            auto histogram = histogramWithCounts(
                std::vector<uint64_t>(8, 0));
            MoEOverlayResidencyAuthority authority({
                .initial_plan = twoTierNodeLocalPlan(owner_order),
                .model_metadata = eightExpertMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .phase_service_profile = twoTierServiceProfile(),
                .migration_cost_profile =
                    threeParticipantMigrationProfile(
                        /*transfer_and_repack_ns=*/1,
                        /*inference_interference_ns=*/1),
                .migration_economy_policy =
                    MoEOverlayMigrationEconomyPolicy{
                        .historical_window_weight = 0,
                        .current_window_weight = 1,
                        .payoff_horizon_tokens = 370,
                        .minimum_net_benefit_ns = 0,
                        .minimum_residency_generations = 0,
                    },
                .shadow_slots_per_endpoint_layer = 1,
                .max_concurrent_cycles = 1,
                .perf_device = "bounded-priority-tiers",
            });

            /*
             * Initial tier 0 is {e0,e1}; the full target is {e6,e7}.
             * A one-cycle BOM must install one swap without changing which
             * expert constitutes that already-scored cycle.
             */
            const auto transaction =
                authority.proposeFromFrozenHistogramWindow(
                    frozenWindow(
                        1,
                        {10, 9, 8, 7, 6, 5, 100, 90}));
            ASSERT_TRUE(transaction.valid());
            ASSERT_FALSE(transaction.empty());
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);
            ASSERT_GE(transaction.migrations.size(), 2u);
            ASSERT_LE(transaction.migrations.size(), 3u)
                << "A two-tier NodeLocal exchange may close through one "
                   "same-priority participant edge";
            ASSERT_TRUE(
                transaction.migration_cycles.front().valid(
                    transaction.migrations));
            EXPECT_TRUE(transaction.economy.enabled);
            EXPECT_GT(
                transaction.economy.projected_net_benefit_ns,
                0u);
            const auto direction_count = [&](const auto direction)
            {
                return std::count_if(
                    transaction.migrations.begin(),
                    transaction.migrations.end(),
                    [&](const auto &migration)
                    { return migration.direction == direction; });
            };
            EXPECT_EQ(
                direction_count(
                    MoEOverlayTierMigrationDirection::Promotion),
                1);
            EXPECT_EQ(
                direction_count(
                    MoEOverlayTierMigrationDirection::Demotion),
                1);
            EXPECT_LE(
                direction_count(
                    MoEOverlayTierMigrationDirection::SamePriority),
                1)
                << "A bounded combined cycle may advance participant "
                   "placement without changing its tier-selected hot expert";
            EXPECT_EQ(
                transaction.candidate->owner_map.ownerFor(0, 6)->tier_idx,
                0);
            EXPECT_EQ(authority.stats().capacity_bounded_proposals, 1u);
            EXPECT_EQ(authority.stats().target_cycles_omitted, 1u);
        }
    }

    /**
     * @brief Bounded tier selection must use endpoint critical-path economy.
     *
     * Expert 2 has the largest individual count, but it sits on a CPU endpoint
     * that is not gating the layer. Promoting it alone leaves participant 3's
     * 180-activation critical path unchanged. Either expert 6 or 7 has a
     * slightly smaller count, yet promoting one of them reduces the actual
     * layer makespan. This is the reduced form of the real 122B failure where
     * raw-count preselection discarded every profitable tier alternative
     * before the measured economy scheduler could inspect it.
     */
    TEST(
        Test__MoEOverlayResidencyAuthority,
        BoundedTierPreselectionRetainsProfitableCriticalPathAlternative)
    {
        ScopedPerfStats perf;
        auto histogram = fourParticipantHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierThreeCpuParticipantPlan(),
            .model_metadata = eightExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(4),
            .migration_cost_profile =
                fourParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 2,
            .perf_device = "critical-path-tier-preselection",
        });

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(
                    1,
                    {0, 0, 100, 0, 0, 0, 90, 90}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_FALSE(transaction.empty())
            << "a non-critical raw-count winner must not hide a profitable "
               "tier alternative";
        EXPECT_GT(transaction.economy.projected_net_benefit_ns, 0u);
        EXPECT_TRUE(std::any_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return migration.crossesTier() &&
                       migration.direction ==
                           MoEOverlayTierMigrationDirection::Promotion &&
                       (migration.expert_id == 6 ||
                        migration.expert_id == 7);
            }))
            << "bounded selection must promote from the endpoint that owns "
               "the measured critical path";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        WholeLayerEvidenceFloorSurvivesMultiTierPartition)
    {
        ScopedPerfStats perf;
        auto histogram = fourParticipantHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierThreeCpuParticipantPlan(),
            .model_metadata = eightExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(4),
            .migration_cost_profile =
                fourParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1001,
                .minimum_improvement_per_mille = 1,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 64,
            },
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "whole-layer-evidence-floor",
        });

        /*
         * The complete routed layer has 240 observations and therefore clears
         * the configured 64-observation evidence floor.  Its apportioned CPU
         * tier contains only 50 of those observations.  Partitioning by tier
         * is an ownership optimization detail and must not silently redefine
         * the production window that the shared Dynamic knob qualifies.
         */
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(
                    1,
                    {100, 90, 30, 15, 2, 1, 1, 1}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        ASSERT_EQ(transaction.migrations.size(), 2u);
        EXPECT_TRUE(std::all_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return !migration.crossesTier() &&
                       migration.direction ==
                           MoEOverlayTierMigrationDirection::SamePriority;
            }));
        EXPECT_EQ(
            authority.stats().participant_rebalance_owner_changes,
            2u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        BoundedDynamicWaveAdmitsBothTierAndParticipantAxes)
    {
        ScopedPerfStats perf;
        auto histogram = fourParticipantHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierThreeCpuParticipantPlan(),
            .model_metadata = eightExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(4),
            .migration_cost_profile =
                fourParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 370,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1001,
                .minimum_improvement_per_mille = 1,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 2,
            .perf_device = "bounded-two-axis-dynamic",
        });

        /*
         * The two hottest experts both begin on CPU participant 1, so the two
         * tier exchanges use only participants 0 and 1. Participants 2 and 3
         * are independently and deliberately skewed, making their profitable
         * same-tier swap a disjoint closed cycle. A two-cycle wave must not let
         * the larger tier-residency scores consume both slots forever.
         */
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(
                    1,
                    {10, 9, 100, 90, 80, 70, 1, 1}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migration_cycles.size(), 2u);
        ASSERT_TRUE(transaction.host_admission.has_value());
        EXPECT_TRUE(transaction.host_admission->valid());
        EXPECT_EQ(
            transaction.host_admission->admitted_physical_cycles,
            2u);
        EXPECT_NE(
            transaction.host_admission->admitted_physical_axes
                .tier_residency +
                transaction.host_admission->admitted_physical_axes.combined,
            0u);
        EXPECT_NE(
            transaction.host_admission->admitted_physical_axes
                    .participant_placement +
                transaction.host_admission->admitted_physical_axes.combined,
            0u);
        const auto proposal_stats = authority.stats();
        EXPECT_GT(proposal_stats.participant_rebalance_owner_changes, 0u)
            << "the adversarial CPU ownership must reach cycle admission";
        EXPECT_EQ(proposal_stats.payoff_rejected_cycles, 0u)
            << "both placement axes are deliberately profitable";
        EXPECT_EQ(transaction.economy.projected_service_gain_ns, 9'224u)
            << "the transaction must price both axes once through the joint "
               "participant critical path";
        EXPECT_EQ(
            transaction.economy.projected_transfer_and_repack_ns,
            2u);
        EXPECT_EQ(
            transaction.economy.projected_inference_interference_ns,
            2u);
        EXPECT_EQ(transaction.economy.projected_net_benefit_ns, 9'220u);

        bool admitted_tier_placement = false;
        bool admitted_participant_rebalance = false;
        for (const auto &cycle : transaction.migration_cycles)
        {
            const bool pure_same_tier = std::all_of(
                cycle.migration_indices.begin(),
                cycle.migration_indices.end(),
                [&](const std::size_t migration_index)
                {
                    return !transaction.migrations[migration_index]
                                .crossesTier();
                });
            admitted_participant_rebalance |= pure_same_tier;
            admitted_tier_placement |= !pure_same_tier;
        }
        EXPECT_TRUE(admitted_tier_placement);
        EXPECT_TRUE(admitted_participant_rebalance)
            << "bounded Dynamic scheduling must make progress on both "
               "independent placement axes";

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto *axis_admission = findRecord(
            records, "cycle_axis_admission");
        ASSERT_NE(axis_admission, nullptr);
        EXPECT_NE(
            axis_admission->tags.at(
                "eligible_participant_placement_cycles"),
            "0");
        EXPECT_NE(
            axis_admission->tags.at(
                "admitted_participant_placement_cycles"),
            "0");
        EXPECT_EQ(
            axis_admission->tags.at("admitted_candidate_cycles"),
            "2")
            << "candidate admission must remain distinct from recomposed physical transfer cycles";
        EXPECT_NE(
            axis_admission->tags.find(
                "admitted_candidate_tier_residency_cycles"),
            axis_admission->tags.end());
        EXPECT_NE(
            axis_admission->tags.find(
                "admitted_candidate_participant_placement_cycles"),
            axis_admission->tags.end());
        EXPECT_NE(
            axis_admission->tags.find(
                "admitted_candidate_combined_cycles"),
            axis_admission->tags.end());
        const auto exclusive_axis_total = [&](const char *prefix)
        {
            return std::stoull(axis_admission->tags.at(
                       std::string(prefix) + "_tier_residency_cycles")) +
                   std::stoull(axis_admission->tags.at(
                       std::string(prefix) +
                       "_participant_placement_cycles")) +
                   std::stoull(axis_admission->tags.at(
                       std::string(prefix) + "_combined_cycles"));
        };
        EXPECT_EQ(
            exclusive_axis_total("admitted_candidate"),
            std::stoull(axis_admission->tags.at(
                "admitted_candidate_cycles")))
            << "combined candidates occupy one exclusive telemetry bucket";
        EXPECT_EQ(
            exclusive_axis_total("admitted"),
            transaction.migration_cycles.size())
            << "combined physical cycles occupy one exclusive telemetry bucket";
        EXPECT_NE(
            axis_admission->tags.find("physical_cycle_recomposition"),
            axis_admission->tags.end());
        EXPECT_EQ(
            axis_admission->tags.at(
                "independent_axis_reservation_active"),
            "true");
    }

    /**
     * @brief A phase-regressive first layer must not hide later alternatives.
     *
     * Multi-tier waves reserve one two-edge participant slot. The raw-count
     * planner deliberately finds a swap on layer zero first, but participant
     * three is so slow there that measured economics must reject it. Layer one
     * has the same skew over equal CPU costs and is profitable. Candidate
     * search therefore has to span layers while final admission remains bound
     * to one participant cycle.
     */
    TEST(
        Test__MoEOverlayResidencyAuthority,
        MultiTierParticipantSearchSkipsPhaseRegressiveFirstLayer)
    {
        ScopedPerfStats perf;
        auto initial_plan = twoTierThreeCpuParticipantPlan();
        auto metadata = eightExpertMetadata();
        metadata.num_layers = 2;

        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = metadata.num_layers;
        histogram_config.num_experts = metadata.num_experts;
        histogram_config.top_k = 2;
        histogram_config.window_size = 4;
        histogram_config.sockets = {
            DeviceId::cuda(0),
            DeviceId::cpu(),
            DeviceId::cpu(),
            DeviceId::cpu(),
        };
        histogram_config.ownership =
            MoELayeredExpertOwnership::uniform(
                metadata.num_layers,
                4,
                {0, 0, 1, 1, 2, 2, 3, 3});
        auto histogram =
            std::make_unique<DecodeExpertHistogram>(histogram_config);

        auto service_profile =
            std::make_shared<MoERoutedTierServiceProfile>();
        service_profile->identity =
            "two-layer-alternative-participant-service-v1";
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            service_profile->costs.push_back({
                .tier_index = 0,
                .layer = layer,
                .nanoseconds_per_activation = {10, 10, 10},
            });
            service_profile->costs.push_back({
                .tier_index = 1,
                .layer = layer,
                .nanoseconds_per_activation = {100, 100, 100},
            });
            for (int participant = 0; participant < 4; ++participant)
            {
                std::uint64_t cost = participant == 0 ? 10u : 100u;
                if (layer == 0 && participant == 3)
                    cost = 1'000u;
                service_profile->participant_costs.push_back({
                    .participant_id = participant,
                    .layer = layer,
                    .nanoseconds_per_activation = {cost, cost, cost},
                });
            }
        }

        auto migration_profile =
            std::make_shared<MoEOverlayMigrationCostProfile>();
        migration_profile->identity =
            "two-layer-alternative-participant-migration-v1";
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            for (int source = 0; source < 4; ++source)
            {
                for (int destination = 0; destination < 4; ++destination)
                {
                    if (source == destination)
                        continue;
                    migration_profile->costs.push_back({
                        .source_participant = source,
                        .destination_participant = destination,
                        .layer = layer,
                        .transfer_and_repack_ns = 1,
                        .inference_interference_ns = 1,
                    });
                }
            }
        }

        MoEOverlayResidencyAuthority authority({
            .initial_plan = std::move(initial_plan),
            .model_metadata = metadata,
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = std::move(service_profile),
            .migration_cost_profile = std::move(migration_profile),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 2'048,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1'000,
                .minimum_improvement_per_mille = 0,
                .maximum_swaps_per_layer = 2,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 2,
            .perf_device = "two-layer-participant-search",
        });

        const std::vector<std::uint64_t> layer_counts{
            10, 9, 100, 90, 80, 70, 1, 1,
        };
        auto window = std::make_shared<DecodeExpertHistogramWindow>();
        window->generation = 1;
        window->num_layers = metadata.num_layers;
        window->num_experts = metadata.num_experts;
        const std::size_t entries =
            static_cast<std::size_t>(metadata.num_layers) *
            static_cast<std::size_t>(metadata.num_experts);
        window->expert_counts.reserve(entries);
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            window->expert_counts.insert(
                window->expert_counts.end(),
                layer_counts.begin(),
                layer_counts.end());
        }
        window->source_expert_counts.assign(
            entries * kExpertHistogramProductionSourceCount,
            0u);
        std::copy(
            window->expert_counts.begin(),
            window->expert_counts.end(),
            window->source_expert_counts.begin());
        window->token_count = std::accumulate(
            window->expert_counts.begin(),
            window->expert_counts.end(),
            std::uint64_t{0});
        window->source_token_counts[0] = window->token_count;
        ASSERT_TRUE(window->valid());

        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(window);
        ASSERT_TRUE(transaction.valid());
        EXPECT_GE(authority.stats().participant_rebalance_checks, 2u)
            << "candidate search must reach the second routed layer";
        EXPECT_TRUE(std::any_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return migration.layer_idx == 1 &&
                       advancesParticipantPlacement(migration.axis);
            }))
            << "the later phase-safe participant alternative must be admitted";
        const std::size_t admitted_participant_objectives =
            static_cast<std::size_t>(std::count_if(
                transaction.migrations.begin(),
                transaction.migrations.end(),
                [](const auto &migration)
                { return advancesParticipantPlacement(migration.axis); }));
        EXPECT_LE(admitted_participant_objectives, 2u)
            << "alternative search must not expand the physical participant slot";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        BoundedDynamicWavePricesParticipantSkewFromPublishedEpoch)
    {
        ScopedPerfStats perf;
        auto histogram = threeParticipantTwelveExpertHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierNodeLocalPlan(
                RoutedExpertOwnerOrder::Ordinal,
                /*accelerator_capacity=*/6),
            .model_metadata = twelveExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(),
            .migration_cost_profile =
                threeParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 1'000,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1001,
                .minimum_improvement_per_mille = 1,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 2,
            .perf_device = "published-epoch-two-axis-dynamic",
        });

        /*
         * The unbounded tier target can place all six active CPU experts on
         * the accelerator and leaves only zero-demand experts in the CPU tier.
         * A two-cycle publication cannot perform all six exchanges, however,
         * so after its first tier exchange the first CPU owner still carries
         * two hot experts while the other owns three much colder experts. A
         * paired swap reduces the live makespan. Pricing against the eventual
         * target incorrectly reports zero participant load and starves that
         * axis forever.
         */
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(
                    1,
                    {0, 0, 0, 0, 0, 0,
                     100, 90, 80, 10, 9, 8}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migration_cycles.size(), 2u);

        bool admitted_tier_placement = false;
        bool admitted_participant_rebalance = false;
        for (const auto &cycle : transaction.migration_cycles)
        {
            const bool pure_same_tier = std::all_of(
                cycle.migration_indices.begin(),
                cycle.migration_indices.end(),
                [&](const std::size_t migration_index)
                {
                    return !transaction.migrations[migration_index]
                                .crossesTier();
                });
            admitted_participant_rebalance |= pure_same_tier;
            admitted_tier_placement |= !pure_same_tier;
        }
        EXPECT_TRUE(admitted_tier_placement);
        EXPECT_TRUE(admitted_participant_rebalance)
            << "participant skew must be priced from the epoch that will "
               "actually remain live after a bounded publication";
        EXPECT_GT(
            authority.stats().participant_rebalance_owner_changes,
            0u);

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto *axis_admission = findRecord(
            records, "cycle_axis_admission");
        ASSERT_NE(axis_admission, nullptr);
        EXPECT_NE(
            axis_admission->tags.at(
                "eligible_participant_placement_cycles"),
            "0");
        EXPECT_EQ(
            axis_admission->tags.at("admitted_candidate_cycles"),
            "2");
        const auto *reserved = findRecord(
            records,
            "tier_cycles_reserved_before_live_participant_axis");
        ASSERT_NE(reserved, nullptr);
        EXPECT_GT(reserved->value, 0.0)
            << "participant skew must be planned after a real tier subset";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ParticipantObjectiveSurvivesAbsorptionIntoTierOnlyPhysicalEdges)
    {
        ScopedPerfStats perf;
        auto histogram = threeParticipantTwelveExpertHistogram();
        MoEOverlayResidencyAuthority authority({
            .initial_plan = twoTierNodeLocalPlan(
                RoutedExpertOwnerOrder::Ordinal,
                /*accelerator_capacity=*/6),
            .model_metadata = twelveExpertMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .phase_service_profile = twoTierServiceProfile(),
            .migration_cost_profile =
                threeParticipantMigrationProfile(
                    /*transfer_and_repack_ns=*/1,
                    /*inference_interference_ns=*/1),
            .migration_economy_policy =
                MoEOverlayMigrationEconomyPolicy{
                    .historical_window_weight = 0,
                    .current_window_weight = 1,
                    .payoff_horizon_tokens = 1'000,
                    .minimum_net_benefit_ns = 0,
                    .minimum_residency_generations = 0,
                },
            .participant_rebalance_policy = {
                .enabled = true,
                .imbalance_threshold_per_mille = 1001,
                .minimum_improvement_per_mille = 1,
                .maximum_swaps_per_layer = 4,
                .maximum_plan_entries_per_wave = 16,
                .minimum_window_activations = 1,
            },
            .shadow_slots_per_endpoint_layer = 2,
            .max_concurrent_cycles = 3,
            .perf_device = "combined-axis-without-same-priority-edge",
        });

        /*
         * Experts 6 and 9 displace accelerator experts 0 and 1. Stable tier
         * assignment initially sends those arrivals to CPU participants 1
         * and 2 respectively. Swapping only the two arrivals reduces the CPU
         * makespan from 200 to 110, so the participant objective changes the
         * cross-tier destinations without creating a same-priority transfer.
         * Marginal economy may admit either closed half first; that bounded
         * cycle must still retain the combined objective.
         */
        const auto transaction =
            authority.proposeFromFrozenHistogramWindow(
                frozenWindow(
                    1,
                    {100, 10, 800, 800, 800, 800,
                     1'000, 50, 50, 900, 0, 0}));
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.migrations.size(), 2u);
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);
        EXPECT_EQ(
            authority.stats().participant_rebalance_owner_changes,
            2u);
        EXPECT_TRUE(std::all_of(
            transaction.migrations.begin(),
            transaction.migrations.end(),
            [](const auto &migration)
            {
                return migration.crossesTier() &&
                       migration.direction !=
                           MoEOverlayTierMigrationDirection::SamePriority &&
                       migration.axis ==
                           MoEOptimizationMovementAxis::Combined;
            }))
            << "participant intent must survive when every physical edge crosses tiers";

        RecordingTransport transport(&authority);
        ASSERT_EQ(
            authority.beginApply(transaction, transport).status,
            MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Preparing);
        EXPECT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Publishing);
        ASSERT_EQ(
            authority.advanceBackground().status,
            MoEOverlayResidencyApplyStatus::Published);

        const auto ledger = authority.movementLedger();
        ASSERT_TRUE(ledger.complete());
        ASSERT_EQ(ledger.edges.size(), transaction.migrations.size());
        EXPECT_TRUE(std::all_of(
            ledger.edges.begin(),
            ledger.edges.end(),
            [](const auto &edge)
            {
                return edge.valid() &&
                       edge.axis == MoEOptimizationMovementAxis::Combined;
            }))
            << "the durable authority ledger must retain both completed objectives";

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto *axis_admission = findRecord(
            records, "cycle_axis_admission");
        ASSERT_NE(axis_admission, nullptr);
        EXPECT_EQ(
            axis_admission->tags.at("admitted_combined_cycles"),
            "1");
        EXPECT_EQ(
            axis_admission->tags.at(
                "admitted_participant_placement_cycles"),
            "0")
            << "combined cycles are one exclusive typed telemetry bucket";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        LiveTicketsContinueAcrossPublicationAndRetireOnMaintenanceWorker)
    {
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::HistogramTieredCache,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        RecordingTransport transport(&authority);
        transport.retirement_admission_pending_polls = 2;
        const auto transaction = authority.proposeFromHistogram();

        auto first_old_lease = authority.tryAcquireTicketSnapshot();
        ASSERT_TRUE(first_old_lease.has_value());
        EXPECT_EQ((*first_old_lease)->epoch, 1u);
        EXPECT_EQ(authority.activeTicketCount(), 1u);

        const auto started = authority.beginApply(transaction, transport);
        EXPECT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started);
        EXPECT_EQ(transport.calls, (std::vector<std::string>{"stage"}));
        EXPECT_EQ(authority.snapshot()->epoch, 1u);

        const std::vector<uint64_t> routes_during_migration{
            0, 0, 0, 0, 7, 0};
        histogram->mergeLayerCounts(
            0,
            routes_during_migration.data(),
            static_cast<int>(routes_during_migration.size()),
            false);
        ASSERT_NE(transaction.histogram_window, nullptr);
        EXPECT_EQ(transaction.histogram_window->activationCount(0, 4), 100u);
        EXPECT_EQ(histogram->activationCount(0, 4), 7u)
            << "Inference evidence after proposal belongs to the next bank";

        auto second_old_lease = authority.tryAcquireTicketSnapshot();
        ASSERT_TRUE(second_old_lease.has_value());
        EXPECT_EQ((*second_old_lease)->epoch, 1u)
            << "Ticket admission remains open while background work runs";
        EXPECT_EQ(authority.activeTicketCount(), 2u);

        const auto preparing = authority.advanceBackground();
        EXPECT_EQ(
            preparing.status,
            MoEOverlayResidencyApplyStatus::Preparing);
        EXPECT_EQ(authority.snapshot()->epoch, 1u);

        const auto publishing = authority.advanceBackground();
        EXPECT_EQ(
            publishing.status,
            MoEOverlayResidencyApplyStatus::Publishing);
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_TRUE(transport.candidate_was_exact_addressable);

        const auto published = authority.advanceBackground();
        EXPECT_EQ(published.status, MoEOverlayResidencyApplyStatus::Published);
        EXPECT_EQ(authority.snapshot()->epoch, 2u);
        EXPECT_EQ(histogram->activationCount(0, 4), 7u)
            << "Asynchronous commit must not reset routes collected in flight";
        EXPECT_EQ(authority.pendingRetirementCount(), 1u);
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{
                      "stage", "prepare", "publish", "authority-published"}));

        auto exact_old_lease = authority.tryAcquireTicketSnapshot(1u);
        ASSERT_TRUE(exact_old_lease.has_value());
        EXPECT_EQ((*exact_old_lease)->epoch, 1u)
            << "A captured ticket may claim its exact old epoch after publication";
        EXPECT_FALSE(authority.tryAcquireTicketSnapshot(3u).has_value());

        auto new_lease = authority.tryAcquireTicketSnapshot();
        ASSERT_TRUE(new_lease.has_value());
        EXPECT_EQ((*new_lease)->epoch, 2u);
        EXPECT_EQ(authority.activeTicketCount(), 4u);

        first_old_lease.reset();
        second_old_lease.reset();
        EXPECT_EQ(authority.activeTicketCount(), 2u);
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{
                      "stage", "prepare", "publish", "authority-published"}))
            << "Final return only drops a lease; it never performs cleanup";

        exact_old_lease.reset();
        EXPECT_EQ(authority.activeTicketCount(), 1u);

        const auto still_retiring = authority.advanceBackground();
        EXPECT_EQ(still_retiring.status, MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(authority.pendingRetirementCount(), 1u);
        EXPECT_EQ(transport.retirement_admission_pending_polls, 0);
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{
                      "stage", "prepare", "publish", "authority-published"}))
            << "Device grace-period polling must not retire or close admission";

        auto delayed_exact_old_lease = authority.tryAcquireTicketSnapshot(1u);
        ASSERT_TRUE(delayed_exact_old_lease.has_value())
            << "A delayed captured ticket remains valid until the device grace period permits admission close";
        EXPECT_EQ((*delayed_exact_old_lease)->epoch, 1u);
        delayed_exact_old_lease.reset();

        const auto idle = authority.advanceBackground();
        EXPECT_EQ(idle.status, MoEOverlayResidencyApplyStatus::Idle);
        EXPECT_EQ(authority.pendingRetirementCount(), 0u);
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{
                      "stage",
                      "prepare",
                      "publish",
                      "authority-published",
                      "retire"}));
        EXPECT_EQ(authority.stats().published_with_old_tickets, 1u);
        EXPECT_FALSE(authority.tryAcquireTicketSnapshot(1u).has_value())
            << "A retired epoch must no longer admit delayed tickets";

        new_lease.reset();
        EXPECT_EQ(authority.activeTicketCount(), 0u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        FailedStagingAbortsWithoutPublishingOrRetiring)
    {
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        RecordingTransport transport(&authority);
        transport.stage_ok = false;

        const auto transaction = authority.proposeFromHistogram();
        const auto started = authority.beginApply(transaction, transport);
        ASSERT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started);
        const auto result = authority.advanceBackground();
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::StageFailed);
        EXPECT_EQ(result.error, "injected stage failure");
        EXPECT_EQ(transport.calls,
                  (std::vector<std::string>{"stage", "abort"}));
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_EQ(authority.stats().stage_failures, 1u);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        ShadowCapacityBackpressureDefersWithoutClosingTicketAdmission)
    {
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        RecordingTransport transport(&authority);
        transport.defer_start = true;

        const auto transaction = authority.proposeFromHistogram();
        const auto deferred = authority.beginApply(transaction, transport);
        EXPECT_EQ(deferred.status, MoEOverlayResidencyApplyStatus::Deferred);
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_EQ(authority.stats().deferred_waves, 1u);

        auto lease = authority.tryAcquireTicketSnapshot();
        ASSERT_TRUE(lease.has_value());
        EXPECT_EQ((*lease)->epoch, 1u);
        lease.reset();

        const auto idle = authority.advanceBackground();
        EXPECT_EQ(idle.status, MoEOverlayResidencyApplyStatus::Idle);
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        PreparedContextRestorationReturnsMultiWaveDynamicPlacementToExactInitialOwners)
    {
        ScopedPerfStats perf;
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Ordinal),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .shadow_slots_per_endpoint_layer = 1,
            .max_concurrent_cycles = 1,
            .perf_device = "prepared-context-restoration",
        });
        RecordingTransport transport(&authority);
        const auto initial = authority.snapshot();
        ASSERT_NE(initial, nullptr);
        ASSERT_TRUE(authority.initialPreparedPlacementPublished());

        const auto publish = [&](const MoEOverlayResidencyTransaction &transaction)
        {
            ASSERT_TRUE(transaction.valid());
            ASSERT_FALSE(transaction.empty());
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);
            ASSERT_EQ(
                authority.beginApply(transaction, transport).status,
                MoEOverlayResidencyApplyStatus::Started);
            ASSERT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Preparing);
            ASSERT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Publishing);
            ASSERT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Published);
            ASSERT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Idle);
        };

        std::size_t live_waves = 0;
        for (std::uint64_t generation = 1; generation <= 4; ++generation)
        {
            const auto transaction =
                authority.proposeFromFrozenHistogramWindow(
                    frozenWindow(
                        generation,
                        {90, 20, 80, 10, 100, 70}));
            if (transaction.empty())
                break;
            publish(transaction);
            ++live_waves;
        }
        ASSERT_GE(live_waves, 2u)
            << "the adversarial histogram must move more than one bounded cycle";
        ASSERT_FALSE(authority.initialPreparedPlacementPublished());
        const auto live_movement_ledger = authority.movementLedger();
        ASSERT_TRUE(live_movement_ledger.complete());
        ASSERT_FALSE(live_movement_ledger.edges.empty());

        std::size_t restoration_waves = 0;
        while (!authority.initialPreparedPlacementPublished())
        {
            const auto restoration =
                authority.proposeInitialPreparedPlacementRestoration();
            ASSERT_EQ(
                restoration.purpose,
                MoEOverlayResidencyTransactionPurpose::
                    PreparedContextRestoration);
            ASSERT_EQ(restoration.histogram_window, nullptr);
            ASSERT_EQ(restoration.histogram_generation, 0u);
            ASSERT_FALSE(restoration.economy.enabled);
            publish(restoration);
            ASSERT_LT(++restoration_waves, 16u)
                << "bounded restoration failed to make monotonic progress";
        }

        const auto restored = authority.snapshot();
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(
            restored->layered_ownership,
            initial->layered_ownership);
        ASSERT_EQ(
            restored->placement_plan->placements.size(),
            initial->placement_plan->placements.size());
        for (std::size_t index = 0;
             index < initial->placement_plan->placements.size();
             ++index)
        {
            EXPECT_EQ(
                restored->placement_plan->placements[index].layer,
                initial->placement_plan->placements[index].layer);
            EXPECT_EQ(
                restored->placement_plan->placements[index]
                    .routed_expert_tier,
                initial->placement_plan->placements[index]
                    .routed_expert_tier);
        }

        const auto stats = authority.stats();
        EXPECT_EQ(stats.committed_waves, live_waves);
        EXPECT_EQ(
            stats.prepared_context_restoration_waves,
            restoration_waves);
        EXPECT_GT(stats.prepared_context_restoration_migrations, 0u);
        EXPECT_EQ(
            stats.prepared_context_restoration_cycles,
            restoration_waves);
        const auto post_restoration_ledger = authority.movementLedger();
        EXPECT_EQ(
            post_restoration_ledger.edges,
            live_movement_ledger.edges)
            << "Prepared-context teardown must not become live optimization evidence";

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        EXPECT_NE(
            findRecord(records, "prepared_context_restoration_edges"),
            nullptr);
        EXPECT_NE(
            findRecord(records, "prepared_context_restoration_waves"),
            nullptr);
        EXPECT_NE(
            findRecord(
                records,
                "prepared_context_restoration_capacity_conservation_certifications"),
            nullptr);
        EXPECT_EQ(
            std::count_if(
                records.begin(),
                records.end(),
                [](const auto &record)
                {
                    return record.domain == "moe_overlay_residency" &&
                           record.name ==
                               "capacity_conservation_certifications";
                }),
            static_cast<std::ptrdiff_t>(live_waves))
            << "Prepared-context restoration must not inflate live-wave certification evidence";
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        PreparedContextRestorationIsIdempotentAtInitialPlacement)
    {
        auto histogram = histogramWithCounts({90, 20, 80, 10, 100, 70});
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::RoutedTierRebalanced,
                RoutedExpertOwnerOrder::Random),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
        });
        RecordingTransport transport(&authority);

        const auto restoration =
            authority.proposeInitialPreparedPlacementRestoration();
        EXPECT_TRUE(restoration.valid());
        EXPECT_TRUE(restoration.empty());
        EXPECT_EQ(
            restoration.purpose,
            MoEOverlayResidencyTransactionPurpose::
                PreparedContextRestoration);
        const auto result = authority.beginApply(restoration, transport);
        EXPECT_EQ(
            result.status,
            MoEOverlayResidencyApplyStatus::DynamicNoMovement);
        EXPECT_TRUE(transport.calls.empty());
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_TRUE(authority.initialPreparedPlacementPublished());
    }

    TEST(
        Test__MoEOverlayResidencyAuthority,
        StaticPolicyPublishesExplicitZeroMovementPerfStats)
    {
        ScopedPerfStats perf;
        MoEOverlayResidencyAuthority authority({
            .initial_plan = threeTierPlan(
                RoutedExpertResidencyPolicy::StaticById,
                RoutedExpertOwnerOrder::Random),
            .model_metadata = modelMetadata(),
            .maintenance_mode = MoERebalanceRuntimeMode::Off,
            .histogram = nullptr,
            .perf_device = "CUDA:0",
        });
        RecordingTransport transport(&authority);

        const auto transaction = authority.proposeFromHistogram();
        ASSERT_TRUE(transaction.empty());
        const auto result = authority.beginApply(transaction, transport);
        EXPECT_EQ(result.status, MoEOverlayResidencyApplyStatus::StaticNoMovement);
        EXPECT_TRUE(transport.calls.empty());
        EXPECT_EQ(authority.snapshot()->epoch, 1u);
        EXPECT_EQ(authority.stats().static_no_movement_checks, 1u);
        EXPECT_EQ(authority.stats().committed_migrations, 0u);

        const auto records = PerfStatsCollector::snapshot(
            {"moe_overlay_residency"});
        const auto *static_record = findRecord(
            records,
            "static_no_movement_checks");
        ASSERT_NE(static_record, nullptr);
        EXPECT_DOUBLE_EQ(static_record->value, 1.0);
        const auto *movement_record = findRecord(
            records,
            "committed_expert_migrations");
        ASSERT_NE(movement_record, nullptr);
        EXPECT_DOUBLE_EQ(movement_record->value, 0.0);
    }

} // namespace llaminar2::test
