/**
 * @file Test__MoEOverlayDeviceControllerFabricCUDAAndROCm.cpp
 * @brief Real-device proof of the topology-wide mapped controller fabric.
 *
 * Two synthetic node-local MPI ranks construct the production POSIX channel
 * concurrently. Rank zero owns two CUDA participants and rank one owns four
 * ROCm participants. Both driver families register their process-local mapping
 * only after the group-owner first-touch rendezvous. Exact-stream copies then
 * prove byte visibility in both directions without a device/stream synchronize.
 * Restoration regressions seed accumulated placement drift, then require every
 * captured repair wave to respect the same per-layer capacity as optimization.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEOverlayDevicePhysicalMovement.h"
#include "execution/moe/MoEOverlayDevicePreparedArrivalInbox.h"
#include "execution/moe/MoEOverlayDeviceTransportProtocol.h"
#include "execution/moe/MoEOverlayDevicePlacementPolicy.h"
#include "execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockMPITopology.h"
#include "utils/ScopedGPUStream.h"

#include "MoEOverlayDeviceRuntimePublicationFixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @return A same-node mock context carrying the shared run namespace. */
        std::shared_ptr<MockMPIContext> context(int rank)
        {
            auto result = std::make_shared<MockMPIContext>(rank, 2);
            result->set_topology(
                MockMPITopology::createSimple(
                    rank, /*world_size=*/2, /*ranks_per_node=*/2));
            return result;
        }

        /** Build one homogeneous rank-local GPU domain. */
        RoutedExpertDomain domain(
            std::string name,
            DeviceType type,
            int count,
            int world_rank)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::RANK_LOCAL;
            result.backend = type == DeviceType::CUDA
                                 ? CollectiveBackendType::NCCL
                                 : CollectiveBackendType::RCCL;
            result.owner_rank = world_rank;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            for (int ordinal = 0; ordinal < count; ++ordinal)
            {
                result.participants.push_back(
                    type == DeviceType::CUDA
                        ? GlobalDeviceAddress::cuda(ordinal)
                        : GlobalDeviceAddress::rocm(ordinal));
            }
            return result;
        }

        /**
         * @return Target 2xCUDA/4xROCm topology with either GPU family continuing.
         * @param continuation_type CUDA or ROCm family owning the logical root.
         */
        std::shared_ptr<const MoEOverlayDeviceControllerTopology> topology(
            DeviceType continuation_type = DeviceType::CUDA)
        {
            if (continuation_type != DeviceType::CUDA &&
                continuation_type != DeviceType::ROCm)
            {
                throw std::invalid_argument(
                    "controller fixture continuation must be CUDA or ROCm");
            }
            const bool cuda_continuation =
                continuation_type == DeviceType::CUDA;
            const DeviceType secondary_type = cuda_continuation
                                                  ? DeviceType::ROCm
                                                  : DeviceType::CUDA;
            const int continuation_count = cuda_continuation ? 2 : 4;
            const int secondary_count = cuda_continuation ? 4 : 2;
            const int continuation_rank = cuda_continuation ? 0 : 1;
            const int secondary_rank = cuda_continuation ? 1 : 0;
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "continuation";
            plan.continuation_domain_spec.domain = "continuation";
            plan.continuation_domain_spec.logical_root_participant = 0;
            plan.shared_expert_domain = "continuation";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.authority_execution =
                MoEOverlayAuthorityExecutionKind::DeviceResident;
            plan.domains = {
                domain(
                    "continuation",
                    continuation_type,
                    continuation_count,
                    continuation_rank),
                domain(
                    "secondary",
                    secondary_type,
                    secondary_count,
                    secondary_rank),
            };
            plan.routed_tiers = {
                {
                    .name = "priority_minus_4",
                    .domain = "continuation",
                    .priority = -4,
                },
                {
                    .name = "priority_23",
                    .domain = "secondary",
                    .priority = 23,
                    .fallback = true,
                },
            };
            std::vector<int> routed_expert_tier(16u, 1);
            const int continuation_experts = cuda_continuation ? 4 : 8;
            std::fill_n(
                routed_expert_tier.begin(), continuation_experts, 0);
            plan.placements = {
                {.layer = 0, .routed_expert_tier = routed_expert_tier},
                {.layer = 1, .routed_expert_tier = routed_expert_tier},
            };
            auto resolved = resolveMoEOverlayDeviceControllerTopology(
                plan,
                MoEExpertOwnerMap::build(plan),
                {.world_rank_node_ids = std::array<int, 2>{7, 7}});
            // Mock topology namespaces are stable across processes. Salt the
            // synthetic topology so a killed prior test cannot own this name.
            resolved.topology_fingerprint ^=
                static_cast<std::uint64_t>(::getpid()) << 17u;
            if (resolved.topology_fingerprint == 0u)
                resolved.topology_fingerprint = 1u;
            return std::make_shared<const MoEOverlayDeviceControllerTopology>(
                std::move(resolved));
        }

        /**
         * Publish deterministic measured costs before a Dynamic policy test.
         *
         * This fixture unlocks the production policy gate without expressing
         * a desired placement, command, or epoch on the host. Calibration has
         * its own state-machine tests; this suite targets device policy/RCU.
         */
        void publishSyntheticEconomy(
            MoEOverlayNodeLocalDeviceControllerFabric &first_rank,
            MoEOverlayNodeLocalDeviceControllerFabric &second_rank,
            const MoEOverlayDeviceControllerTopology &resolved,
            std::uint32_t layer_count,
            ExpertHistogramProductionSourceMask economy_sources =
                kAllExpertHistogramProductionSources,
            std::array<std::uint64_t, 3> phase_cost_multipliers = {1u, 1u, 1u},
            std::uint64_t transfer_cost_ns = 1u)
        {
            std::uint32_t tier_count = 0u;
            for (const auto &participant : resolved.participants)
            {
                tier_count = std::max(
                    tier_count,
                    static_cast<std::uint32_t>(participant.tier_idx + 1));
            }
            auto service = std::make_shared<MoERoutedTierServiceProfile>();
            service->identity = "device-controller-test-service-v1";
            service->production_topology = ExpertHistogramProductionTopology(
                std::vector<ExpertHistogramProductionSourceMask>(
                    layer_count, kAllExpertHistogramProductionSources),
                std::vector<ExpertHistogramProductionSourceMask>(
                    layer_count, economy_sources));
            for (std::uint32_t tier = 0u; tier < tier_count; ++tier)
            {
                for (std::uint32_t layer = 0u;
                     layer < layer_count;
                     ++layer)
                {
                    const std::uint64_t cost = 10u + tier * 90u;
                    service->costs.push_back({
                        .tier_index = static_cast<int>(tier),
                        .layer = static_cast<int>(layer),
                        .nanoseconds_per_activation = {
                            economy_sources[0] ? cost * phase_cost_multipliers[0] : 0u,
                            economy_sources[1] ? cost * phase_cost_multipliers[1] : 0u,
                            economy_sources[2] ? cost * phase_cost_multipliers[2] : 0u},
                    });
                }
            }

            auto migration =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            migration->identity = "device-controller-test-migration-v1";
            for (std::uint32_t source = 0u;
                 source < resolved.participants.size();
                 ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < resolved.participants.size();
                     ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u;
                         layer < layer_count;
                         ++layer)
                    {
                        migration->costs.push_back({
                            .source_participant =
                                static_cast<int>(source),
                            .destination_participant =
                                static_cast<int>(destination),
                            .layer = static_cast<int>(layer),
                            .transfer_and_repack_ns = transfer_cost_ns,
                            .inference_interference_ns = 0u,
                        });
                    }
                }
            }
            const MoEOverlayCertifiedEconomyProfiles profiles{
                .service = std::move(service),
                .migration = std::move(migration),
                .policy = {
                    .historical_window_weight = 0u,
                    .current_window_weight = 1u,
                    .payoff_horizon_tokens = 65'536u,
                    .minimum_net_benefit_ns = 0u,
                    .minimum_residency_generations = 0u,
                },
            };
            ASSERT_TRUE(profiles.valid());
            auto &publisher = first_rank.ownsEconomyPublication()
                                  ? first_rank
                                  : second_rank;
            ASSERT_TRUE(publisher.ownsEconomyPublication());
            publisher.publishCertifiedEconomyProfiles(profiles);
            ASSERT_TRUE(first_rank.economyProfilesPublished());
            ASSERT_TRUE(second_rank.economyProfilesPublished());
        }

        /** Poll one exact event without blocking the device or stream. */
        bool await(
            IBackend *backend,
            void *event,
            int ordinal,
            std::chrono::steady_clock::duration timeout)
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (backend && event &&
                   std::chrono::steady_clock::now() < deadline)
            {
                bool ready = false;
                if (!backend->queryEvent(event, ordinal, &ready))
                    return false;
                if (ready)
                    return true;
                std::this_thread::yield();
            }
            return false;
        }

        /**
         * Write one mapped range on a producer and consume it through another
         * backend's alias into ordinary VRAM, using exact terminal events only.
         */
        bool proveVisibility(
            IBackend *producer,
            int producer_ordinal,
            void *producer_alias,
            IBackend *consumer,
            int consumer_ordinal,
            const void *consumer_alias,
            int byte_pattern,
            std::string *error)
        {
            constexpr std::size_t kBytes = 64u;
            void *producer_stream = producer->createStream(producer_ordinal);
            void *consumer_stream = consumer->createStream(consumer_ordinal);
            void *producer_event = producer->createEvent(producer_ordinal);
            void *consumer_event = consumer->createEvent(consumer_ordinal);
            void *consumer_vram = consumer->allocate(kBytes, consumer_ordinal);
            const auto cleanup = [&]
            {
                if (producer_event)
                    producer->destroyEvent(producer_event, producer_ordinal);
                if (consumer_event)
                    consumer->destroyEvent(consumer_event, consumer_ordinal);
                if (producer_stream)
                    producer->destroyStream(
                        producer_stream, producer_ordinal);
                if (consumer_stream)
                    consumer->destroyStream(
                        consumer_stream, consumer_ordinal);
                if (consumer_vram)
                    consumer->free(consumer_vram, consumer_ordinal);
            };
            if (!producer_stream || !consumer_stream || !producer_event ||
                !consumer_event || !consumer_vram)
            {
                if (error)
                    *error = "could not allocate explicit streams/events/VRAM";
                cleanup();
                return false;
            }
            if (!producer->memset(
                    producer_alias,
                    byte_pattern,
                    kBytes,
                    producer_ordinal,
                    producer_stream) ||
                !producer->recordEvent(
                    producer_event,
                    producer_ordinal,
                    producer_stream) ||
                !await(
                    producer,
                    producer_event,
                    producer_ordinal,
                    std::chrono::seconds(5)))
            {
                if (error)
                    *error = "mapped producer write did not complete";
                cleanup();
                return false;
            }
            if (!consumer->deviceCopyAsync(
                    consumer_vram,
                    consumer_alias,
                    kBytes,
                    consumer_ordinal,
                    consumer_stream) ||
                !consumer->recordEvent(
                    consumer_event,
                    consumer_ordinal,
                    consumer_stream) ||
                !await(
                    consumer,
                    consumer_event,
                    consumer_ordinal,
                    std::chrono::seconds(5)))
            {
                if (error)
                    *error = "mapped consumer read did not complete";
                cleanup();
                return false;
            }

            std::array<std::uint8_t, kBytes> observed{};
            if (!consumer->deviceToHost(
                    observed.data(),
                    consumer_vram,
                    observed.size(),
                    consumer_ordinal,
                    consumer_stream))
            {
                if (error)
                    *error = "terminal evidence copy failed";
                cleanup();
                return false;
            }
            const bool matches = std::all_of(
                observed.begin(),
                observed.end(),
                [byte_pattern](std::uint8_t value)
                {
                    return value == static_cast<std::uint8_t>(byte_pattern);
                });
            if (!matches && error)
                *error = "cross-backend mapped bytes disagreed";
            cleanup();
            return matches;
        }

        /** Terminal device evidence for one complete cross-vendor epoch. */
        struct ControllerTransactionEvidence
        {
            MoEOverlayDeviceControllerSharedHeader controller;
            MoEOverlayDeviceControllerCommandHeader command;
            MoEOverlayDeviceControllerPolicyResult policy;
            std::vector<MoEOverlayDeviceMovementCommand> commands;
            std::array<MoEOverlayDeviceControllerGroupRecord, 2> groups{};
            std::uint32_t transport_groups_acquired = 0u;
            std::uint32_t transport_groups_prepared = 0u;
            std::uint32_t transport_groups_published = 0u;
            std::uint32_t transport_groups_retired = 0u;
            std::uint32_t transport_groups_restored = 0u;
            float leader_elapsed_ms = 0.0F;
        };

        /** @return Canonical digest consumed by the device command validator. */
        std::uint64_t commandDigest(
            const MoEOverlayDeviceMovementCommand *entries,
            std::uint32_t count)
        {
            std::uint64_t digest = moeOverlayCommandDigestSeed(count);
            constexpr std::size_t kWordsPerEntry =
                sizeof(MoEOverlayDeviceMovementCommand) /
                sizeof(std::uint64_t);
            const auto *words = reinterpret_cast<const std::uint64_t *>(
                entries);
            for (std::size_t index = 0u;
                 index < static_cast<std::size_t>(count) * kWordsPerEntry;
                 ++index)
            {
                digest ^= moeOverlayCommandDigestWord(words[index], index);
            }
            return digest;
        }

        /** Build a two-layer adversarial snapshot for both placement axes. */
        MoEOverlayDevicePlacementPolicyInput adversarialPolicyInput()
        {
            MoEOverlayDevicePlacementPolicyInput input;
            input.num_layers = 2u;
            input.num_experts = 16u;
            for (std::uint32_t participant = 0u; participant < 6u;
                 ++participant)
            {
                input.participants.push_back({
                    .participant_id = participant,
                    .tier_priority = participant < 2u ? -4 : 23,
                    .tier_index = participant < 2u ? 0 : 1,
                });
            }
            input.collected_state.assign(6u * 2u * 16u, 0u);
            const std::array<std::uint32_t, 16> owners = {
                2u, 3u, 4u, 5u, 0u, 0u, 1u, 1u,
                2u, 2u, 3u, 3u, 4u, 4u, 5u, 5u,
            };
            const std::array<std::uint64_t, 16> descending = {
                1600u, 1500u, 1400u, 1300u,
                1200u, 1100u, 1000u, 900u,
                800u, 700u, 600u, 500u,
                40u, 30u, 20u, 10u,
            };
            for (std::uint32_t layer = 0u; layer < input.num_layers; ++layer)
            {
                for (std::uint32_t expert = 0u;
                     expert < input.num_experts;
                     ++expert)
                {
                    const std::uint32_t owner = owners[expert];
                    const std::uint64_t count =
                        layer == 0u
                            ? descending[expert]
                            : descending[input.num_experts - 1u - expert];
                    const std::size_t offset =
                        (static_cast<std::size_t>(owner) * input.num_layers +
                         layer) *
                            input.num_experts +
                        expert;
                    input.collected_state[offset] =
                        moe_rebalance_policy::packCollectedState(
                            count,
                            /*active_transfer_slots=*/0u,
                            /*physically_resident=*/true,
                            /*transfer_backed=*/false,
                            /*authoritative_owner=*/true);
                }
            }
            input.payload_bytes_per_layer = {4096u, 6144u};
            input.base_epoch = 3u;
            input.minimum_window_activations = 1u;
            input.maximum_cycles_per_wave = 32u;
            input.command_capacity = 64u;
            MoEOverlayDevicePlacementEconomyInput economy;
            economy.tier_count = 2u;
            economy.routed_experts_per_token = 1u;
            economy.transaction_generation = 1u;
            const std::size_t plane_words =
                static_cast<std::size_t>(input.num_layers) *
                input.num_experts;
            economy.phase_expert_demand.assign(
                kMoEOverlayDeviceControllerDemandPhaseCount * plane_words,
                0u);
            for (std::uint32_t layer = 0u;
                 layer < input.num_layers;
                 ++layer)
            {
                for (std::uint32_t expert = 0u;
                     expert < input.num_experts;
                     ++expert)
                {
                    std::uint64_t demand = 0u;
                    for (std::uint32_t participant = 0u;
                         participant < input.participants.size();
                         ++participant)
                    {
                        demand += moe_rebalance_policy::
                            collectedStateActivationCount(
                                input.collected_state[
                                    (static_cast<std::size_t>(participant) *
                                         input.num_layers +
                                     layer) *
                                        input.num_experts +
                                    expert]);
                    }
                    economy.phase_expert_demand[
                        (static_cast<std::size_t>(
                             kMoEOverlayDeviceControllerEconomyPrefillPhase) *
                             input.num_layers +
                         layer) *
                            input.num_experts +
                        expert] = demand;
                }
            }
            economy.service_costs.assign(
                static_cast<std::size_t>(economy.tier_count) *
                    input.num_layers *
                    kMoEOverlayDeviceControllerEconomyServicePhaseCount,
                0u);
            for (std::uint32_t tier = 0u; tier < economy.tier_count; ++tier)
            {
                for (std::uint32_t layer = 0u;
                     layer < input.num_layers;
                     ++layer)
                {
                    for (std::uint32_t phase = 0u;
                         phase <
                             kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                         ++phase)
                    {
                        economy.service_costs[
                            (static_cast<std::size_t>(tier) * input.num_layers +
                             layer) *
                                kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                            phase] = 10u + tier * 90u;
                    }
                }
            }
            economy.migration_costs.assign(
                input.participants.size() * input.participants.size() *
                    input.num_layers,
                {});
            for (std::uint32_t source = 0u;
                 source < input.participants.size();
                 ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < input.participants.size();
                     ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u;
                         layer < input.num_layers;
                         ++layer)
                    {
                        economy.migration_costs[
                            (static_cast<std::size_t>(source) *
                                 input.participants.size() +
                             destination) *
                                input.num_layers +
                            layer] = {
                            .transfer_and_repack_ns = 1u,
                            .inference_interference_ns = 0u,
                        };
                    }
                }
            }
            economy.last_moved_generation.assign(
                plane_words,
                kMoEOverlayDeviceControllerNeverMovedGeneration);
            economy.payoff_horizon_tokens = 65'536u;
            input.economy = std::move(economy);
            return input;
        }

        /**
         * @brief Build the exact two-slot policy shape that once starved skew.
         *
         * Layer zero has two high-value tier exchanges and no participant
         * correction. Layer one is tier-optimal and has one lower-value closed
         * cycle that independently reduces the priority-23 makespan. A
         * two-cycle wave therefore proves the device selector searches the
         * complete layer cursor before spending a reserved slot on fallback.
         */
        MoEOverlayDevicePlacementPolicyInput boundedTwoAxisPolicyInput()
        {
            auto input = adversarialPolicyInput();
            input.num_layers = 2u;
            input.num_experts = 12u;
            input.collected_state.assign(
                6u * input.num_layers * input.num_experts, 0u);
            const std::array<std::uint32_t, 12> layer_zero_owners = {
                2u, 3u, 1u, 0u, 0u, 1u,
                4u, 5u, 5u, 4u, 3u, 2u,
            };
            const std::array<std::uint32_t, 12> layer_one_owners = {
                0u, 1u, 1u, 0u, 2u, 3u,
                5u, 4u, 5u, 4u, 3u, 2u,
            };
            const std::array<std::uint64_t, 12> counts = {
                1200u, 1100u, 1000u, 900u,
                850u, 825u, 800u, 700u,
                600u, 500u, 100u, 50u,
            };
            for (std::uint32_t layer = 0u;
                 layer < input.num_layers;
                 ++layer)
            {
                const auto &owners = layer == 0u
                                         ? layer_zero_owners
                                         : layer_one_owners;
                for (std::uint32_t expert = 0u;
                     expert < input.num_experts;
                     ++expert)
                {
                    input.collected_state[
                        (static_cast<std::size_t>(owners[expert]) *
                             input.num_layers +
                         layer) *
                            input.num_experts +
                        expert] = moe_rebalance_policy::packCollectedState(
                        counts[expert],
                        /*active_transfer_slots=*/0u,
                        /*physically_resident=*/true,
                        /*transfer_backed=*/false,
                        /*authoritative_owner=*/true);
                }
            }
            input.payload_bytes_per_layer = {4096u, 4096u};
            input.maximum_cycles_per_wave = 2u;
            input.dynamic_maximum_cycles_per_layer = 2u;
            input.dynamic_imbalance_threshold_per_mille = 1000u;
            input.dynamic_minimum_improvement_per_mille = 0u;
            input.command_capacity = input.num_experts;

            auto &economy = input.economy.value();
            const std::size_t plane_words =
                static_cast<std::size_t>(input.num_layers) *
                input.num_experts;
            economy.phase_expert_demand.assign(
                kMoEOverlayDeviceControllerDemandPhaseCount * plane_words,
                0u);
            for (std::uint32_t layer = 0u;
                 layer < input.num_layers;
                 ++layer)
            {
                for (std::uint32_t expert = 0u;
                     expert < input.num_experts;
                     ++expert)
                {
                    economy.phase_expert_demand[
                        static_cast<std::size_t>(
                            kMoEOverlayDeviceControllerEconomyPrefillPhase) *
                            plane_words +
                        static_cast<std::size_t>(layer) * input.num_experts +
                        expert] = counts[expert];
                }
            }
            economy.service_costs.assign(
                static_cast<std::size_t>(economy.tier_count) *
                    input.num_layers *
                    kMoEOverlayDeviceControllerEconomyServicePhaseCount,
                0u);
            for (std::uint32_t tier = 0u; tier < economy.tier_count; ++tier)
            {
                for (std::uint32_t layer = 0u;
                     layer < input.num_layers;
                     ++layer)
                {
                    for (std::uint32_t phase = 0u;
                         phase <
                             kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                         ++phase)
                    {
                        economy.service_costs[
                            (static_cast<std::size_t>(tier) *
                                 input.num_layers +
                             layer) *
                                kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                            phase] = 10u + tier * 90u;
                    }
                }
            }
            economy.migration_costs.assign(
                input.participants.size() * input.participants.size() *
                    input.num_layers,
                {});
            for (std::uint32_t source = 0u;
                 source < input.participants.size();
                 ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < input.participants.size();
                     ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u;
                         layer < input.num_layers;
                         ++layer)
                    {
                        economy.migration_costs[
                            (static_cast<std::size_t>(source) *
                                 input.participants.size() +
                             destination) *
                                input.num_layers +
                            layer] = {
                            .transfer_and_repack_ns = 1u,
                            .inference_interference_ns = 0u,
                        };
                    }
                }
            }
            economy.last_moved_generation.assign(
                plane_words,
                kMoEOverlayDeviceControllerNeverMovedGeneration);
            return input;
        }

        /** Extract the exact layer-major initial owner table from a snapshot. */
        std::vector<std::uint32_t> initialOwners(
            const MoEOverlayDevicePlacementPolicyInput &input)
        {
            std::vector<std::uint32_t> owners(
                static_cast<std::size_t>(input.num_layers) *
                    input.num_experts,
                std::numeric_limits<std::uint32_t>::max());
            for (std::uint32_t layer = 0u; layer < input.num_layers; ++layer)
            {
                for (std::uint32_t expert = 0u;
                     expert < input.num_experts;
                     ++expert)
                {
                    std::uint32_t owner_count = 0u;
                    for (std::uint32_t participant = 0u;
                         participant < input.participants.size();
                         ++participant)
                    {
                        const auto word = input.collected_state[
                            (static_cast<std::size_t>(participant) *
                                 input.num_layers +
                             layer) *
                                input.num_experts +
                            expert];
                        if (moe_rebalance_policy::
                                collectedStateAuthoritativeOwner(word))
                        {
                            owners[static_cast<std::size_t>(layer) *
                                       input.num_experts +
                                   expert] = participant;
                            ++owner_count;
                        }
                    }
                    if (owner_count != 1u)
                    {
                        throw std::logic_error(
                            "test snapshot does not have exactly one initial owner");
                    }
                }
            }
            return owners;
        }

        /** Publish one group's complete setup-only differential snapshot. */
        bool seedGroupSnapshot(
            IBackend *backend,
            const MoEOverlayDeviceControllerParticipantBinding &root,
            const MoEOverlayDeviceControllerFabricGroupLayout &group,
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::string *error)
        {
            if (!backend || !root.valid() || !root.group_root ||
                group.group_id != static_cast<std::uint32_t>(root.group_id) ||
                input.collected_state.empty())
            {
                if (error)
                    *error = "snapshot fixture received an invalid group root";
                return false;
            }
            std::vector<std::uint64_t> words;
            words.reserve(static_cast<std::size_t>(group.collected_state_words));
            for (std::uint32_t member = 0u;
                 member < group.participant_count;
                 ++member)
            {
                const std::uint32_t participant = group.participant_ids[member];
                for (std::uint32_t phase = 0u;
                     phase < kMoEOverlayDeviceControllerDemandPhaseCount; ++phase)
                {
                for (std::uint32_t layer = 0u; layer < input.num_layers; ++layer)
                {
                    for (std::uint32_t expert = 0u;
                         expert < input.num_experts;
                         ++expert)
                    {
                        const auto word = input.collected_state[
                            (static_cast<std::size_t>(participant) *
                                 input.num_layers +
                             layer) *
                                input.num_experts +
                            expert];
                        // Legacy synthetic policy input describes decode demand;
                        // every plane still carries identical ownership metadata.
                        words.push_back(phase == 0u ? word :
                            word & ~moe_rebalance_policy::kCollectedStateActivationCountMask);
                    }
                }
                }
            }
            if (words.size() != group.collected_state_words)
            {
                if (error)
                    *error = "snapshot fixture disagrees with mapped group geometry";
                return false;
            }

            void *stream = backend->createStream(root.device.ordinal);
            void *event = backend->createEvent(root.device.ordinal);
            const bool submitted = stream && event && backend->hostToDevice(
                root.group_collected_state,
                words.data(),
                words.size() * sizeof(std::uint64_t),
                root.device.ordinal,
                stream) &&
                backend->recordEvent(event, root.device.ordinal, stream);
            const bool complete = submitted && await(
                backend,
                event,
                root.device.ordinal,
                std::chrono::seconds(5));
            if (event)
                backend->destroyEvent(event, root.device.ordinal);
            if (stream)
                backend->destroyStream(stream, root.device.ordinal);
            if (!complete && error)
                *error = "device snapshot fixture publication did not complete";
            return complete;
        }

        /** @return Every participant binding in immutable global-id order. */
        std::vector<MoEOverlayDeviceControllerParticipantBinding>
        allParticipantBindings(
            const MoEOverlayNodeLocalDeviceControllerFabric &cuda_rank,
            const MoEOverlayNodeLocalDeviceControllerFabric &rocm_rank)
        {
            std::vector<MoEOverlayDeviceControllerParticipantBinding> result;
            result.reserve(6u);
            for (const int participant : cuda_rank.localParticipantIds())
                result.push_back(cuda_rank.participantBinding(participant));
            for (const int participant : rocm_rank.localParticipantIds())
                result.push_back(rocm_rank.participantBinding(participant));
            std::sort(
                result.begin(),
                result.end(),
                [](const auto &left, const auto &right)
                {
                    return left.participant_id < right.participant_id;
                });
            return result;
        }

        /**
         * @return Every locally owned group-root transport lane in group order.
         *
         * Discovering roots from participant metadata keeps the proof valid if
         * either backend owns the continuation domain or if a later topology
         * adds more same-node groups.
         */
        std::vector<MoEOverlayDeviceControllerTransportBinding>
        allTransportBindings(
            const MoEOverlayNodeLocalDeviceControllerFabric &cuda_rank,
            const MoEOverlayNodeLocalDeviceControllerFabric &rocm_rank)
        {
            std::vector<MoEOverlayDeviceControllerTransportBinding> result;
            const std::array<const MoEOverlayNodeLocalDeviceControllerFabric *,
                             2>
                fabrics = {&cuda_rank, &rocm_rank};
            for (const auto *fabric : fabrics)
            {
                for (const int participant : fabric->localParticipantIds())
                {
                    const auto binding =
                        fabric->participantBinding(participant);
                    if (binding.group_root)
                        result.push_back(
                            fabric->transportBinding(binding.group_id));
                }
            }
            std::sort(
                result.begin(),
                result.end(),
                [](const auto &left, const auto &right)
                {
                    return left.group_id < right.group_id;
                });
            return result;
        }

        /** Own one minimal but ABI-real model runtime table on an exact GPU. */
        class ControllerRuntimeFixture final
            : public IMoEOverlayDeviceInitialRuntimePublisher
        {
        public:
            /** Allocate and publish two live runtime layers for one participant. */
            ControllerRuntimeFixture(
                IBackend *backend,
                DeviceId device,
                std::int32_t overlay_participant,
                std::uint32_t domain_participant,
                std::uint32_t domain_participant_count)
                : backend_(backend), device_(device)
            {
                if (!backend_ || !device_.is_gpu() ||
                    domain_participant_count == 0u ||
                    domain_participant >= domain_participant_count)
                {
                    throw std::invalid_argument(
                        "controller runtime fixture requires one valid GPU participant");
                }

                constexpr std::uint32_t kLayers = 2u;
                constexpr std::uint32_t kExperts = 16u;
                constexpr std::uint32_t kTopK = 2u;
                std::array<DeviceMoELayerRuntime, kLayers> host_layers{};
                for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                {
                    auto &runtime = host_layers[layer];
                    runtime.active_bank = 0u;
                    runtime.active_epoch = 3u;
                    runtime.expert_count = kExperts;
                    runtime.top_k = kTopK;
                    runtime.participant_id = domain_participant;
                    runtime.participant_count = domain_participant_count;
                    auto &bank = runtime.banks[0];
                    bank.epoch = 3u;
                    bank.expert_count = kExperts;
                    for (std::uint32_t expert = 0u;
                         expert < kExperts;
                         ++expert)
                    {
                        const std::uint32_t owner =
                            expert % domain_participant_count;
                        auto &descriptor = bank.experts[expert];
                        descriptor.logical_expert_id =
                            static_cast<std::int32_t>(expert);
                        descriptor.owner_participant =
                            static_cast<std::int32_t>(owner);
                        bank.resident_participant_mask[expert] = 1u << owner;
                        if (owner == domain_participant)
                        {
                            descriptor.flags = toMoEExpertFlags(
                                DeviceMoEExpertFlags::Valid |
                                DeviceMoEExpertFlags::Resident);
                            runtime.decode_local_histogram[expert] =
                                static_cast<std::uint64_t>(
                                    overlay_participant + 1) *
                                static_cast<std::uint64_t>(
                                    (layer + 1u) * (expert + 1u));
                        }
                    }
                }

                runtime_layers_ = static_cast<DeviceMoELayerRuntime *>(
                    backend_->allocate(
                        sizeof(host_layers), device_.ordinal));
                void *const stream = backend_->createStream(device_.ordinal);
                void *const terminal = backend_->createEvent(device_.ordinal);
                const bool submitted = runtime_layers_ && stream && terminal &&
                    backend_->hostToDevice(
                        runtime_layers_,
                        host_layers.data(),
                        sizeof(host_layers),
                        device_.ordinal,
                        stream) &&
                    backend_->recordEvent(
                        terminal, device_.ordinal, stream);
                const bool complete = submitted && await(
                    backend_,
                    terminal,
                    device_.ordinal,
                    std::chrono::seconds(5));
                if (terminal)
                    backend_->destroyEvent(terminal, device_.ordinal);
                if (stream)
                    backend_->destroyStream(stream, device_.ordinal);
                if (!complete)
                {
                    if (runtime_layers_)
                    {
                        backend_->free(runtime_layers_, device_.ordinal);
                        runtime_layers_ = nullptr;
                    }
                    throw std::runtime_error(
                        "controller runtime fixture device publication failed");
                }

                binding_ = {
                    .device = device_,
                    .runtime_layers_device = runtime_layers_,
                    .overlay_participant_id = overlay_participant,
                    .domain_participant_id = domain_participant,
                    .domain_participant_count = domain_participant_count,
                    .layer_count = kLayers,
                    .expert_count = kExperts,
                    .top_k = kTopK,
                    .initial_runtime_publisher = this,
                };
            }

            /**
             * The fixture synchronously certifies its setup upload above, so
             * no additional device edge is needed when a retained test
             * controller adopts it.
             */
            [[nodiscard]] bool publishMoEOverlayDeviceInitialRuntime(
                void *controller_stream) override
            {
                return controller_stream != nullptr;
            }

            /** Release the table after every retained graph has been destroyed. */
            ~ControllerRuntimeFixture()
            {
                if (runtime_layers_)
                    backend_->free(runtime_layers_, device_.ordinal);
            }

            ControllerRuntimeFixture(const ControllerRuntimeFixture &) = delete;
            ControllerRuntimeFixture &operator=(
                const ControllerRuntimeFixture &) = delete;

            /** @return Stable graph-capture binding for this fixture. */
            const MoEOverlayDeviceControllerRuntimeBinding &binding()
                const noexcept
            {
                return binding_;
            }

        private:
            IBackend *backend_ = nullptr;
            DeviceId device_ = DeviceId::invalid();
            DeviceMoELayerRuntime *runtime_layers_ = nullptr;
            MoEOverlayDeviceControllerRuntimeBinding binding_;
        };

        /**
         * Exercise one complete controller transaction without an intermediate
         * host policy wait, copy, or policy-state read.
         *
         * The leader queues its complete dependency chain first. Kernels that
         * depend on the ROCm group spin only inside their tiny mapped-control
         * block, leaving the host free to submit the follower chain. Both GPU
         * streams then advance through system-scope release/acquire edges.
         * Independent group-local workers consume only the immutable command
         * bytes and publish physical readiness; they never read histograms or
         * author policy. The caller finally observes all participant terminal
         * events and copies diagnostic evidence.
         */
        bool runControllerTransaction(
            IBackend *cuda,
            IBackend *rocm,
            const std::vector<MoEOverlayDeviceControllerParticipantBinding>
                &participant_bindings,
            const std::vector<MoEOverlayDeviceControllerTransportBinding>
                &transport_bindings,
            const MoEOverlayDeviceControllerFabricLayout &layout,
            MoEOverlayDeviceControllerTransactionKind kind,
            std::uint64_t base_epoch,
            ControllerTransactionEvidence *evidence,
            std::string *error,
            bool inject_destination_collision = false)
        {
            using llaminar::v2::kernels::KernelFactory;

            if (error)
                error->clear();
            if (!cuda || !rocm || participant_bindings.empty() || !evidence)
            {
                if (error)
                    *error = "controller transaction received incomplete resources";
                return false;
            }
            *evidence = {};

            // Reject malformed authority topology before allocating any GPU
            // resource. This also makes the cleanup path independent of which
            // participant happened to be visited first.
            const auto leader_count = std::count_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [](const auto &binding)
                {
                    return binding.authority_leader;
                });
            const bool transport_geometry_valid =
                transport_bindings.size() == layout.groups.size() &&
                std::all_of(
                    transport_bindings.begin(),
                    transport_bindings.end(),
                    [](const auto &binding)
                    {
                        return binding.valid();
                    }) &&
                std::adjacent_find(
                    transport_bindings.begin(),
                    transport_bindings.end(),
                    [](const auto &left, const auto &right)
                    {
                        return left.group_id >= right.group_id;
                    }) == transport_bindings.end();
            if (leader_count != 1 || !transport_geometry_valid)
            {
                if (error)
                {
                    *error = leader_count != 1
                                 ? "controller transaction requires exactly one authority leader"
                                 : "controller transaction received incomplete or unordered transport roots";
                }
                return false;
            }

            struct Endpoint
            {
                MoEOverlayDeviceControllerParticipantBinding binding;
                IBackend *backend = nullptr;
                std::unique_ptr<IMoEKernel> kernel;
                void *stream = nullptr;
                void *terminal = nullptr;
            };

            std::vector<Endpoint> endpoints;
            endpoints.reserve(participant_bindings.size());
            Endpoint *leader = nullptr;
            for (const auto &binding : participant_bindings)
            {
                if (!binding.valid())
                {
                    if (error)
                        *error = "controller transaction received an invalid participant binding";
                    return false;
                }
                IBackend *const backend =
                    binding.device.type == DeviceType::CUDA ? cuda :
                    binding.device.type == DeviceType::ROCm ? rocm : nullptr;
                if (!backend)
                {
                    if (error)
                        *error = "controller transaction received a non-GPU participant";
                    return false;
                }
                endpoints.push_back({
                    .binding = binding,
                    .backend = backend,
                    .kernel = KernelFactory::createMoEKernel(binding.device),
                    .stream = backend->createStream(binding.device.ordinal),
                    .terminal = binding.authority_leader
                                    ? backend->createTimingEvent(
                                          binding.device.ordinal)
                                    : backend->createEvent(
                                          binding.device.ordinal),
                });
                if (binding.authority_leader)
                {
                    if (leader)
                    {
                        if (error)
                            *error = "controller transaction received multiple authority leaders";
                        return false;
                    }
                    leader = &endpoints.back();
                }
            }

            void *leader_start = leader
                                     ? leader->backend->createTimingEvent(
                                           leader->binding.device.ordinal)
                                     : nullptr;
            void *policy_device = leader
                                      ? leader->backend->allocate(
                                            sizeof(
                                                MoEOverlayDeviceControllerPolicyResult),
                                            leader->binding.device.ordinal)
                                      : nullptr;
            const auto cleanup = [&]
            {
                if (leader_start && leader)
                {
                    leader->backend->destroyEvent(
                        leader_start, leader->binding.device.ordinal);
                }
                if (policy_device && leader)
                {
                    leader->backend->free(
                        policy_device, leader->binding.device.ordinal);
                }
                for (auto &endpoint : endpoints)
                {
                    endpoint.kernel.reset();
                    if (endpoint.terminal)
                    {
                        endpoint.backend->destroyEvent(
                            endpoint.terminal,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.stream)
                    {
                        endpoint.backend->destroyStream(
                            endpoint.stream,
                            endpoint.binding.device.ordinal);
                    }
                }
            };
            const bool endpoints_complete =
                leader && leader_start && policy_device &&
                std::all_of(
                    endpoints.begin(),
                    endpoints.end(),
                    [](const Endpoint &endpoint)
                    {
                        return endpoint.kernel && endpoint.stream &&
                               endpoint.terminal;
                    });
            if (!endpoints_complete)
            {
                if (error)
                    *error = "could not allocate explicit controller streams, events, or policy storage";
                cleanup();
                return false;
            }

            const bool static_check =
                kind == MoEOverlayDeviceControllerTransactionKind::StaticCheck;
            const bool dynamic =
                kind ==
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement;
            std::array<MoEOverlayDeviceMovementCommand, 2> command_entries{};
            command_entries[0] = {
                .op = static_cast<std::uint32_t>(
                    dynamic
                        ? MoEOverlayDeviceMovementOp::DurableMove
                        : MoEOverlayDeviceMovementOp::TransientAssignment),
                .ordinal = 0u,
                .layer = 0u,
                .expert = 1u,
                .source_participant = 0u,
                .destination_participant = dynamic ? 2u : 0u,
                .payload_slot = dynamic ? 0u : kMoEOverlayDeviceInvalidSlot,
                .flags = static_cast<std::uint32_t>(
                    dynamic
                        ? MoEOverlayDeviceMovementAxis::TierResidency
                        : MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                .payload_bytes = dynamic ? 4096u : 0u,
                .source_epoch = base_epoch,
                .candidate_epoch = dynamic ? base_epoch + 1u : base_epoch,
            };
            command_entries[1] = command_entries[0];
            command_entries[1].ordinal = 1u;
            command_entries[1].expert =
                inject_destination_collision ? 1u : 2u;
            command_entries[1].payload_slot = 1u;
            const std::uint32_t command_count =
                static_check ? 0u : (inject_destination_collision ? 2u : 1u);
            MoEOverlayDeviceControllerPolicyResult policy{
                .kind = static_cast<std::uint32_t>(kind),
                .command_count = command_count,
                .command_digest = commandDigest(
                    command_count == 0u ? nullptr : command_entries.data(),
                    command_count),
                .packed_weight_bytes =
                    dynamic ? 4096u * command_count : 0u,
            };
            if (!leader->backend->hostToDevice(
                     policy_device,
                     &policy,
                     sizeof(policy),
                     leader->binding.device.ordinal,
                     leader->stream) ||
                (command_count != 0u &&
                 !leader->backend->hostToDevice(
                     leader->binding.command_entries,
                     command_entries.data(),
                     sizeof(command_entries[0]) * command_count,
                     leader->binding.device.ordinal,
                     leader->stream)) ||
                !leader->backend->recordEvent(
                    leader_start,
                    leader->binding.device.ordinal,
                    leader->stream))
            {
                if (error)
                    *error = "could not publish the setup-only policy fixture";
                cleanup();
                return false;
            }

            const auto enqueue = [kind, policy_device](
                                     Endpoint &endpoint,
                                     MoEOverlayDeviceControllerAction action)
            {
                const bool consumes_policy =
                    action == MoEOverlayDeviceControllerAction::PublishCommand ||
                    action == MoEOverlayDeviceControllerAction::AuthorDynamicPolicy;
                return endpoint.kernel &&
                       endpoint.kernel->runMoEOverlayDeviceControllerAction(
                           {.stream = endpoint.stream},
                           {
                               .binding = endpoint.binding.deviceBinding(),
                               .action = action,
                               .transaction_kind =
                                   action == MoEOverlayDeviceControllerAction::
                                                 BeginTransaction
                                       ? kind
                                       : MoEOverlayDeviceControllerTransactionKind::
                                             Invalid,
                               .demand_phase =
                                   action == MoEOverlayDeviceControllerAction::
                                                 BeginTransaction &&
                                           kind ==
                                               MoEOverlayDeviceControllerTransactionKind::
                                                   DynamicPlacement
                                       ? MoEOverlayDeviceDemandPhase::Prefill
                                       : MoEOverlayDeviceDemandPhase::Invalid,
                               .policy_result =
                                   consumes_policy
                                       ? static_cast<
                                             MoEOverlayDeviceControllerPolicyResult *>(
                                             policy_device)
                                       : nullptr,
                           });
            };

            // Each transport lane retains its own consumed cursor. Read only
            // that host-owned cursor before the authority publishes the next
            // immutable command; the worker never needs a controller shadow.
            std::vector<std::uint64_t> prior_transport_transactions;
            prior_transport_transactions.reserve(transport_bindings.size());
            for (const auto &binding : transport_bindings)
            {
                prior_transport_transactions.push_back(
                    std::atomic_ref<std::uint64_t>(
                        binding.transport->command_transaction)
                        .load(std::memory_order_acquire));
            }

            bool submitted =
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::BeginTransaction) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::
                        PublishParticipantSnapshot) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::PublishGroupSnapshot);
            submitted = submitted &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::PublishCommand) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::AcknowledgePrepared) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::BeginCommit) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::AcknowledgePublished) &&
                enqueue(
                    *leader,
                    MoEOverlayDeviceControllerAction::PublishAdmission);
            if (dynamic)
            {
                submitted = submitted &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::BeginDynamicRetirement) &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::AcknowledgeRetired) &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::CompleteDynamicRetirement);
            }
            else if (!static_check)
            {
                submitted = submitted &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::BeginLLEPRestore) &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::AcknowledgeLLEPRestored) &&
                            enqueue(
                                *leader,
                                MoEOverlayDeviceControllerAction::CompleteLLEPRestore);
            }
            submitted = submitted && leader->backend->recordEvent(
                                             leader->terminal,
                                             leader->binding.device.ordinal,
                                             leader->stream);

            // Submit every independent participant only after the complete
            // leader chain is queued. A synchronous launch on either backend
            // would deadlock here, making this a direct regression guard for
            // hidden host waits and serialized cross-device dispatch.
            for (auto &endpoint : endpoints)
            {
                if (&endpoint == leader)
                    continue;
                submitted = submitted && enqueue(
                    endpoint,
                    MoEOverlayDeviceControllerAction::
                        PublishParticipantSnapshot);
                if (endpoint.binding.group_root)
                {
                    submitted = submitted &&
                                enqueue(
                                    endpoint,
                                    MoEOverlayDeviceControllerAction::
                                        PublishGroupSnapshot) &&
                                enqueue(
                                    endpoint,
                                    MoEOverlayDeviceControllerAction::
                                        AcknowledgePrepared) &&
                                enqueue(
                                    endpoint,
                                    MoEOverlayDeviceControllerAction::
                                        AcknowledgePublished);
                    if (dynamic)
                    {
                        submitted = submitted && enqueue(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AcknowledgeRetired);
                    }
                    else if (!static_check)
                    {
                        submitted = submitted && enqueue(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AcknowledgeLLEPRestored);
                    }
                }
                submitted = submitted &&
                            enqueue(
                                endpoint,
                                MoEOverlayDeviceControllerAction::
                                    AwaitTransactionComplete) &&
                            endpoint.backend->recordEvent(
                                endpoint.terminal,
                                endpoint.binding.device.ordinal,
                                endpoint.stream);
            }

            struct TransportWorkerEvidence
            {
                bool acquired = false;
                bool prepared = false;
                bool published = false;
                bool retired = false;
                bool restored = false;
                std::string error;
            };
            std::vector<TransportWorkerEvidence> transport_results(
                transport_bindings.size());
            std::vector<std::jthread> transport_workers;
            transport_workers.reserve(transport_bindings.size());

            // Invalid-command coverage must fail before command publication;
            // no physical worker is allowed to bless those bytes. Every valid
            // transaction, including zero-byte Static/LLEP, is nevertheless
            // acquired so each lane's cursor remains exact across epochs.
            if (submitted && !inject_destination_collision)
            {
                for (std::size_t index = 0u;
                     index < transport_bindings.size();
                     ++index)
                {
                    transport_workers.emplace_back(
                        [binding = transport_bindings[index],
                         after = prior_transport_transactions[index],
                         &result = transport_results[index]]
                        {
                            MoEOverlayDeviceTransportProtocol protocol(binding);
                            const auto deadline =
                                std::chrono::steady_clock::now() +
                                std::chrono::seconds(5);
                            MoEOverlayDeviceTransportAcquireResult acquired;
                            while (std::chrono::steady_clock::now() < deadline)
                            {
                                acquired = protocol.tryAcquire(after);
                                if (acquired.status !=
                                    MoEOverlayDeviceTransportAcquireStatus::
                                        Waiting)
                                {
                                    break;
                                }
                                std::this_thread::sleep_for(
                                    std::chrono::microseconds(50));
                            }
                            if (acquired.status !=
                                MoEOverlayDeviceTransportAcquireStatus::Ready)
                            {
                                result.error = acquired.error.empty()
                                                   ? "timed out acquiring the device-authored command"
                                                   : acquired.error;
                                protocol.fail();
                                return;
                            }
                            result.acquired = true;
                            if (!acquired.batch.movesWeights())
                                return;

                            // Keep the readiness edge observably asynchronous:
                            // every GPU chain was already submitted, and each
                            // group progresses this independent worker in
                            // parallel before the device-owned commit opens.
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(2));
                            if (!protocol.publishPrepared(
                                    acquired.batch, &result.error))
                            {
                                return;
                            }
                            result.prepared = true;

                            const auto wait_until = [deadline](auto predicate)
                            {
                                while (std::chrono::steady_clock::now() <
                                       deadline)
                                {
                                    if (predicate())
                                        return true;
                                    std::this_thread::sleep_for(
                                        std::chrono::microseconds(50));
                                }
                                return false;
                            };
                            if (!wait_until(
                                    [&]
                                    {
                                        return protocol.commitRequested(
                                            acquired.batch);
                                    }))
                            {
                                result.error =
                                    "timed out waiting for the device-owned commit edge";
                                protocol.fail();
                                return;
                            }
                            if (!protocol.publishPublished(
                                    acquired.batch, &result.error))
                            {
                                return;
                            }
                            result.published = true;

                            const auto transaction_kind = static_cast<
                                MoEOverlayDeviceControllerTransactionKind>(
                                acquired.batch.header.kind);
                            if (transaction_kind ==
                                MoEOverlayDeviceControllerTransactionKind::
                                    DynamicPlacement)
                            {
                                if (!wait_until(
                                        [&]
                                        {
                                            return protocol.retirementRequested(
                                                acquired.batch);
                                        }))
                                {
                                    result.error =
                                        "timed out waiting for device-owned durable retirement";
                                    protocol.fail();
                                    return;
                                }
                                if (!protocol.publishRetired(
                                        acquired.batch, &result.error))
                                {
                                    return;
                                }
                                result.retired = true;
                            }
                            else if (
                                transaction_kind ==
                                MoEOverlayDeviceControllerTransactionKind::
                                    CurrentBatchLLEP)
                            {
                                if (!wait_until(
                                        [&]
                                        {
                                            return protocol.restorationRequested(
                                                acquired.batch);
                                        }))
                                {
                                    result.error =
                                        "timed out waiting for device-owned LLEP restoration";
                                    protocol.fail();
                                    return;
                                }
                                if (!protocol.publishRestored(
                                        acquired.batch, &result.error))
                                {
                                    return;
                                }
                                result.restored = true;
                            }
                        });
                }
            }

            bool complete = submitted;
            for (const auto &endpoint : endpoints)
            {
                if (submitted)
                {
                    complete = await(
                                   endpoint.backend,
                                   endpoint.terminal,
                                   endpoint.binding.device.ordinal,
                                   std::chrono::seconds(5)) &&
                               complete;
                }
            }
            for (auto &worker : transport_workers)
                worker.join();
            for (const auto &result : transport_results)
            {
                evidence->transport_groups_acquired += result.acquired ? 1u : 0u;
                evidence->transport_groups_prepared += result.prepared ? 1u : 0u;
                evidence->transport_groups_published += result.published ? 1u : 0u;
                evidence->transport_groups_retired += result.retired ? 1u : 0u;
                evidence->transport_groups_restored += result.restored ? 1u : 0u;
                if (!result.error.empty())
                {
                    complete = false;
                    if (error)
                    {
                        if (!error->empty())
                            *error += "; ";
                        *error += result.error;
                    }
                }
            }
            if (!complete)
            {
                if (error && error->empty())
                    *error = submitted
                                 ? "cross-vendor controller transaction did not reach every participant terminal"
                                 : "a controller action was rejected during asynchronous submission";
                cleanup();
                return false;
            }

            bool copied = leader->backend->eventElapsedTimeMs(
                              leader_start,
                              leader->terminal,
                              leader->binding.device.ordinal,
                              &evidence->leader_elapsed_ms) &&
                          leader->backend->deviceToHost(
                              &evidence->controller,
                              leader->binding.controller,
                              sizeof(evidence->controller),
                              leader->binding.device.ordinal,
                              leader->stream) &&
                          leader->backend->deviceToHost(
                              &evidence->command,
                              leader->binding.command,
                              sizeof(evidence->command),
                              leader->binding.device.ordinal,
                              leader->stream) &&
                          leader->backend->deviceToHost(
                              &evidence->policy,
                              policy_device,
                              sizeof(evidence->policy),
                              leader->binding.device.ordinal,
                              leader->stream);
            evidence->commands.resize(evidence->command.command_count);
            if (copied && !evidence->commands.empty())
            {
                copied = leader->backend->deviceToHost(
                    evidence->commands.data(),
                    leader->binding.command_entries,
                    evidence->commands.size() *
                        sizeof(MoEOverlayDeviceMovementCommand),
                    leader->binding.device.ordinal,
                    leader->stream);
            }
            for (std::size_t group = 0u;
                 copied && group < evidence->groups.size(); ++group)
            {
                const auto offset = static_cast<std::size_t>(
                    layout.groups.at(group).group_record_offset);
                copied = leader->backend->deviceToHost(
                    &evidence->groups[group],
                    leader->binding.mapped_base_device + offset,
                    sizeof(evidence->groups[group]),
                    leader->binding.device.ordinal,
                    leader->stream);
            }
            if (!copied && error)
                *error = "terminal controller evidence copy failed";
            cleanup();
            return copied;
        }

        /** Terminal proof for one topology-wide inactive-bank construction. */
        struct RuntimeCandidateApplyEvidence
        {
            MoEOverlayDeviceControllerSharedHeader controller;
            MoEOverlayDeviceControllerCommandHeader command;
            MoEOverlayDeviceControllerPolicyResult policy;
            std::vector<MoEOverlayDeviceMovementCommand> commands;
            /** Terminal copy of the device-owned committed hysteresis ledger. */
            std::vector<std::uint64_t> economy_last_moved;
            std::vector<MoEOverlayDeviceRuntimeApplyStatus> apply_statuses;
            std::vector<std::vector<DeviceMoELayerRuntime>> runtime_layers;
            std::vector<DeviceMoEOverlayEpochControl> epoch_controls;
            std::vector<DeviceMoEOverlayEpochStatus> epoch_statuses;
            std::vector<MoEOverlayDeviceControllerParticipantRecord>
                participant_records;
            std::uint32_t transport_groups_prepared = 0u;
            std::uint32_t transport_groups_published = 0u;
            std::uint32_t transport_groups_retired = 0u;
            std::uint32_t old_epoch_readers_acquired = 0u;
            std::uint32_t new_epoch_readers_completed = 0u;
            std::uint32_t retirement_readiness_deferrals_observed = 0u;
            std::uint32_t old_epoch_readers_released = 0u;
            /** Deterministic E+1 acquisitions overlapping E reclamation. */
            std::uint32_t post_readiness_acquisition_guards = 0u;
        };

        /** Explicit adversarial mutation applied to one otherwise real wave. */
        enum class RuntimeCandidateApplyFault
        {
            None,
            OmitFirstDestinationArrival,
        };

        /** Durable device-authority objective exercised by the publication rig. */
        enum class RuntimeCandidateApplyObjective
        {
            DynamicPlacement,
            PreparedContextRestore,
        };

        /** Typed execution policy for the runtime-publication integration rig. */
        struct RuntimeCandidateApplyOptions
        {
            bool complete_epoch = false;
            /** Select the retained Dynamic observation-only continuation. */
            bool expect_no_movement = false;
            /** Hold old readers across publication and prove new admission runs. */
            bool exercise_reader_grace_period = false;
            RuntimeCandidateApplyFault fault =
                RuntimeCandidateApplyFault::None;
            /** Select ordinary optimization or terminal prepared-owner repair. */
            RuntimeCandidateApplyObjective objective =
                RuntimeCandidateApplyObjective::DynamicPlacement;
        };

        /**
         * Build every local inactive bank from one device-authored Dynamic wave.
         *
         * All participant maintenance streams are submitted before any host
         * observation. One background worker per group authenticates the
         * immutable command, submits pinned descriptor uploads to every member,
         * waits only on those transfer events, and release-publishes readiness.
         */
        bool runRuntimeCandidateApply(
            IBackend *cuda,
            IBackend *rocm,
            const MoEOverlayDeviceControllerTopology &topology,
            const std::vector<MoEOverlayDeviceControllerParticipantBinding>
                &participant_bindings,
            const std::vector<MoEOverlayDeviceControllerTransportBinding>
                &transport_bindings,
            std::vector<std::unique_ptr<
                MoEOverlayDeviceRuntimePublicationFixture>> &fixtures,
            RuntimeCandidateApplyEvidence *evidence,
            std::string *error,
            RuntimeCandidateApplyOptions options = {})
        {
            using llaminar::v2::kernels::KernelFactory;

            if (error)
                error->clear();
            if (!cuda || !rocm || !evidence || participant_bindings.empty() ||
                fixtures.size() != participant_bindings.size())
            {
                if (error)
                    *error = "runtime apply received incomplete resources";
                return false;
            }
            *evidence = {};
            const bool prepared_context_restore =
                options.objective ==
                RuntimeCandidateApplyObjective::PreparedContextRestore;
            const auto transaction_kind = prepared_context_restore
                ? MoEOverlayDeviceControllerTransactionKind::
                      PreparedContextRestore
                : MoEOverlayDeviceControllerTransactionKind::
                      DynamicPlacement;
            const auto demand_phase = prepared_context_restore
                ? MoEOverlayDeviceDemandPhase::Invalid
                : MoEOverlayDeviceDemandPhase::Prefill;
            const auto author_action = prepared_context_restore
                ? MoEOverlayDeviceControllerAction::
                      AuthorPreparedContextRestore
                : MoEOverlayDeviceControllerAction::AuthorDynamicPolicy;
            const std::uint64_t initial_epoch =
                transport_bindings.empty()
                    ? 0u
                    : transport_bindings.front()
                          .controller->current_durable_epoch;
            const std::uint64_t prior_transaction =
                transport_bindings.empty()
                    ? 0u
                    : transport_bindings.front()
                          .controller->command_transaction;
            if (initial_epoch == 0u)
            {
                if (error)
                    *error = "runtime apply has no initial durable epoch";
                return false;
            }

            const auto fixture_for = [&fixtures](int participant)
                -> MoEOverlayDeviceRuntimePublicationFixture *
            {
                const auto found = std::find_if(
                    fixtures.begin(),
                    fixtures.end(),
                    [participant](const auto &fixture)
                    {
                        return fixture &&
                               fixture->participantId() == participant;
                    });
                return found == fixtures.end() ? nullptr : found->get();
            };

            struct Endpoint
            {
                MoEOverlayDeviceControllerParticipantBinding binding;
                MoEOverlayDeviceRuntimePublicationFixture *fixture = nullptr;
                IBackend *backend = nullptr;
                std::unique_ptr<IMoEKernel> kernel;
                DeviceMoERebalanceConfig snapshot_config;
                void *stream = nullptr;
                void *terminal = nullptr;
                void *reader_stream = nullptr;
                void *reader_event = nullptr;
                void *admission_event = nullptr;
                void *retirement_event = nullptr;
            };
            std::vector<Endpoint> endpoints;
            endpoints.reserve(participant_bindings.size());
            Endpoint *leader = nullptr;
            for (const auto &binding : participant_bindings)
            {
                IBackend *const backend =
                    binding.device.type == DeviceType::CUDA ? cuda :
                    binding.device.type == DeviceType::ROCm ? rocm : nullptr;
                auto *const fixture = fixture_for(binding.participant_id);
                if (!binding.valid() || !backend || !fixture ||
                    !fixture->runtimeBinding().publicationValid())
                {
                    if (error)
                        *error = "runtime apply participant binding is incomplete";
                    return false;
                }
                const auto &runtime = fixture->runtimeBinding();
                endpoints.push_back({
                    .binding = binding,
                    .fixture = fixture,
                    .backend = backend,
                    .kernel = KernelFactory::createMoEKernel(binding.device),
                    .snapshot_config = DeviceMoERebalanceConfig{
                        .num_layers = runtime.layer_count,
                        .num_experts = runtime.expert_count,
                        .top_k = runtime.top_k,
                        .participant_id = runtime.domain_participant_id,
                        .participant_count = runtime.domain_participant_count,
                        .root_participant = 0u,
                        .window_size_tokens = 1u,
                        .layer_window_start = 0u,
                        .layer_window_count = runtime.layer_count,
                        .layer_wave_count = runtime.layer_count,
                    },
                    .stream = backend->createStream(binding.device.ordinal),
                    .terminal = backend->createEvent(binding.device.ordinal),
                    .reader_stream = options.exercise_reader_grace_period
                                         ? backend->createStream(
                                               binding.device.ordinal)
                                         : nullptr,
                    .reader_event = options.exercise_reader_grace_period
                                        ? backend->createEvent(
                                              binding.device.ordinal)
                                        : nullptr,
                    .admission_event =
                        options.exercise_reader_grace_period &&
                                binding.authority_leader
                            ? backend->createEvent(binding.device.ordinal)
                            : nullptr,
                    .retirement_event = options.exercise_reader_grace_period
                                            ? backend->createEvent(
                                                  binding.device.ordinal)
                                            : nullptr,
                });
                if (binding.authority_leader)
                    leader = &endpoints.back();
            }

            void *policy_device = leader
                                      ? leader->backend->allocate(
                                            sizeof(
                                                MoEOverlayDeviceControllerPolicyResult),
                                            leader->binding.device.ordinal)
                                      : nullptr;
            // The publication rig uses the same finite retained author epoch
            // as serving. Destroy its executable before releasing any embedded
            // result pointer or stream at the terminal boundary.
            std::unique_ptr<IGPUGraphCapture> author_graph;
            const auto cleanup = [&]
            {
                author_graph.reset();
                if (policy_device && leader)
                {
                    leader->backend->free(
                        policy_device, leader->binding.device.ordinal);
                }
                for (auto &endpoint : endpoints)
                {
                    endpoint.kernel.reset();
                    if (endpoint.terminal)
                    {
                        endpoint.backend->destroyEvent(
                            endpoint.terminal,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.reader_event)
                    {
                        endpoint.backend->destroyEvent(
                            endpoint.reader_event,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.retirement_event)
                    {
                        endpoint.backend->destroyEvent(
                            endpoint.retirement_event,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.admission_event)
                    {
                        endpoint.backend->destroyEvent(
                            endpoint.admission_event,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.reader_stream)
                    {
                        endpoint.backend->destroyStream(
                            endpoint.reader_stream,
                            endpoint.binding.device.ordinal);
                    }
                    if (endpoint.stream)
                    {
                        endpoint.backend->destroyStream(
                            endpoint.stream,
                            endpoint.binding.device.ordinal);
                    }
                }
            };
            if (options.exercise_reader_grace_period &&
                !options.complete_epoch)
            {
                if (error)
                    *error = "reader grace-period proof requires a complete epoch";
                cleanup();
                return false;
            }
            if (!leader || !policy_device ||
                !std::all_of(
                    endpoints.begin(),
                    endpoints.end(),
                    [require_reader = options.exercise_reader_grace_period](
                        const Endpoint &endpoint)
                    {
                        return endpoint.kernel && endpoint.stream &&
                               endpoint.terminal &&
                               (!require_reader ||
                                (endpoint.reader_stream &&
                                 endpoint.reader_event &&
                                 endpoint.retirement_event &&
                                 (!endpoint.binding.authority_leader ||
                                  endpoint.admission_event))) &&
                               validateDeviceMoERebalanceConfig(
                                   endpoint.snapshot_config);
                    }))
            {
                if (error)
                    *error = "runtime apply could not allocate endpoint resources";
                cleanup();
                return false;
            }

            const auto enqueue_action = [policy_device,
                                         transaction_kind,
                                         demand_phase,
                                         author_action](
                                            Endpoint &endpoint,
                                            MoEOverlayDeviceControllerAction action)
            {
                const bool policy_action =
                    action ==
                        MoEOverlayDeviceControllerAction::AuthorDynamicPolicy ||
                    action == MoEOverlayDeviceControllerAction::
                                  AuthorPreparedContextRestore ||
                    action ==
                        MoEOverlayDeviceControllerAction::PublishCommand;
                return endpoint.kernel->runMoEOverlayDeviceControllerAction(
                    {.stream = endpoint.stream},
                    {
                        .binding = endpoint.binding.deviceBinding(),
                        .action = action,
                        .transaction_kind =
                            action ==
                                    MoEOverlayDeviceControllerAction::
                                        BeginTransaction
                                ? transaction_kind
                                : MoEOverlayDeviceControllerTransactionKind::
                                      Invalid,
                        .demand_phase =
                            action ==
                                    MoEOverlayDeviceControllerAction::
                                        BeginTransaction ||
                                    action == author_action
                                ? demand_phase
                                : MoEOverlayDeviceDemandPhase::Invalid,
                        .policy_result = policy_action
                                             ? static_cast<
                                                   MoEOverlayDeviceControllerPolicyResult *>(
                                                   policy_device)
                                             : nullptr,
                        .runtime_publication =
                            action ==
                                    MoEOverlayDeviceControllerAction::
                                        ApplyRuntimeCandidate ||
                                    action ==
                                        MoEOverlayDeviceControllerAction::
                                            PublishRuntimeCandidate ||
                                    action ==
                                    MoEOverlayDeviceControllerAction::
                                            PublishRuntimeRetirement
                                ? endpoint.fixture->publicationBinding()
                                : MoEOverlayDeviceRuntimePublicationBinding{},
                        .retirement_readiness =
                            action ==
                                    MoEOverlayDeviceControllerAction::
                                        PublishRuntimeRetirementReadiness
                                ? MoEOverlayDeviceRetirementReadinessBinding{
                                      .epoch_control = endpoint.fixture
                                                           ->runtimeBinding()
                                                           .epoch_control}
                                : MoEOverlayDeviceRetirementReadinessBinding{},
                    });
            };
            const auto enqueue_snapshot = [](Endpoint &endpoint)
            {
                const auto plane_words = static_cast<std::size_t>(
                    endpoint.snapshot_config.num_layers) * endpoint.snapshot_config.num_experts;
                for (std::uint32_t phase = 0u;
                     phase < kMoEOverlayDeviceControllerDemandPhaseCount; ++phase)
                {
                    if (!endpoint.kernel->packDeviceRebalanceHistograms(
                            {.stream = endpoint.stream},
                            endpoint.fixture->runtimeBinding().runtime_layers_device,
                            endpoint.binding.participant_collected_state + phase * plane_words,
                            endpoint.snapshot_config, nullptr, nullptr, 1u, 1u << phase))
                        return false;
                }
                return true;
            };
            const auto enqueue_reader_acquire = [](Endpoint &endpoint,
                                                   std::uint32_t slot)
            {
                return endpoint.kernel->acquireMoEOverlayEpoch(
                    {.stream = endpoint.reader_stream},
                    endpoint.fixture->runtimeBinding().epoch_control,
                    endpoint.fixture->requestTicket(slot),
                    endpoint.fixture->requestStatus(slot),
                    &endpoint.binding.controller->admission_epoch);
            };
            const auto enqueue_reader_release = [](Endpoint &endpoint,
                                                   std::uint32_t slot)
            {
                return endpoint.kernel->releaseMoEOverlayEpoch(
                    {.stream = endpoint.reader_stream},
                    endpoint.fixture->runtimeBinding().epoch_control,
                    endpoint.fixture->requestTicket(slot),
                    endpoint.fixture->requestStatus(slot));
            };
            const auto append_error = [error](std::string message)
            {
                if (!error)
                    return;
                if (!error->empty())
                    *error += "; ";
                *error += std::move(message);
            };
            const auto await_reader_wave = [&endpoints, &append_error](
                                                const char *phase)
            {
                bool ready = true;
                for (const auto &endpoint : endpoints)
                {
                    // Poll every independently submitted device even if one
                    // fails, so teardown never abandons a live reader stream.
                    const bool endpoint_ready = await(
                        endpoint.backend,
                        endpoint.reader_event,
                        endpoint.binding.device.ordinal,
                        std::chrono::seconds(5));
                    if (!endpoint_ready)
                    {
                        append_error(
                            std::string(phase ? phase : "reader") +
                            " event timed out for participant " +
                            std::to_string(
                                endpoint.binding.participant_id));
                    }
                    ready = endpoint_ready && ready;
                }
                return ready;
            };
            const auto await_retirement_wave =
                [&endpoints, &append_error](const char *phase)
            {
                bool ready = true;
                for (const auto &endpoint : endpoints)
                {
                    const bool endpoint_ready = await(
                        endpoint.backend,
                        endpoint.retirement_event,
                        endpoint.binding.device.ordinal,
                        std::chrono::seconds(5));
                    if (!endpoint_ready)
                    {
                        append_error(
                            std::string(phase ? phase : "retirement") +
                            " event timed out for participant " +
                            std::to_string(
                                endpoint.binding.participant_id));
                    }
                    ready = endpoint_ready && ready;
                }
                return ready;
            };
            // Submit dependency wavefronts across every participant. HIP may
            // apply backpressure when a later command is enqueued behind a
            // live mapped-memory wait. Giving every producer its turn before
            // group-root waits makes that impossible without introducing a
            // host observation or serial synchronization.
            bool captured = false;
            auto &author_worker = GPUDeviceContextPool::instance().getContext(
                leader->binding.device);
            author_worker.submitAndWait([&]
            {
                author_graph = author_worker.createGraphCapture(leader->stream);
                if (!author_graph || !author_graph->beginCapture())
                    return;
                const bool nodes = enqueue_action(*leader, author_action) &&
                    enqueue_action(*leader, MoEOverlayDeviceControllerAction::PublishCommand) &&
                    enqueue_action(*leader, MoEOverlayDeviceControllerAction::CompleteEmptyDynamicDecision);
                // End capture even when a launch was rejected so its scope
                // cannot leak into cleanup or another test on this worker.
                const bool ended = author_graph->endCapture();
                captured = nodes && ended && author_graph->instantiate();
            });
            if (!captured)
            {
                append_error("runtime apply could not capture its complete author epoch");
                cleanup();
                return false;
            }
            bool submitted = enqueue_action(
                *leader,
                MoEOverlayDeviceControllerAction::BeginTransaction);
            for (auto &endpoint : endpoints)
            {
                submitted = submitted && enqueue_snapshot(endpoint);
                submitted = submitted && enqueue_action(
                    endpoint,
                    MoEOverlayDeviceControllerAction::
                        PublishParticipantSnapshot);
            }
            for (auto &endpoint : endpoints)
            {
                if (endpoint.binding.group_root)
                {
                    submitted = submitted && enqueue_action(
                        endpoint,
                        MoEOverlayDeviceControllerAction::
                            PublishGroupSnapshot);
                }
            }
            submitted = submitted && author_graph->launchOnStream(leader->stream);

            struct TransportResult
            {
                bool transfers_submitted = false;
                bool moves_weights = false;
                bool prepared = false;
                bool published = false;
                bool retired = false;
                std::string error;
            };
            std::vector<TransportResult> transport_results(
                transport_bindings.size());
            std::vector<std::jthread> workers;
            workers.reserve(transport_bindings.size());
            if (submitted)
            {
                for (std::size_t index = 0u;
                     index < transport_bindings.size();
                     ++index)
                {
                    workers.emplace_back(
                        [binding = transport_bindings[index],
                         &topology,
                         &fixture_for,
                         options,
                         prior_transaction,
                         &result = transport_results[index]]
                        {
                            MoEOverlayDeviceTransportProtocol protocol(binding);
                            const auto deadline =
                                std::chrono::steady_clock::now() +
                                std::chrono::seconds(10);
                            MoEOverlayDeviceTransportAcquireResult acquired;
                            while (std::chrono::steady_clock::now() < deadline)
                            {
                                acquired = protocol.tryAcquire(
                                    prior_transaction);
                                if (acquired.status !=
                                    MoEOverlayDeviceTransportAcquireStatus::
                                        Waiting)
                                {
                                    break;
                                }
                                std::this_thread::sleep_for(
                                    std::chrono::microseconds(50));
                            }
                            if (acquired.status !=
                                MoEOverlayDeviceTransportAcquireStatus::Ready)
                            {
                                result.error = acquired.error.empty()
                                                   ? "timed out acquiring Dynamic command"
                                                   : acquired.error;
                                protocol.fail();
                                return;
                            }
                            std::atomic_ref<bool>(result.moves_weights)
                                .store(
                                    acquired.batch.movesWeights(),
                                    std::memory_order_release);
                            if (!acquired.batch.movesWeights())
                            {
                                // A zero-movement Dynamic command has no
                                // descriptor DMA or physical completion edge.
                                // Its retained continuation is still selected
                                // from the authenticated command shape.
                                std::atomic_ref<bool>(
                                    result.transfers_submitted)
                                    .store(true, std::memory_order_release);
                                return;
                            }
                            const auto group_it = std::find_if(
                                topology.groups.begin(),
                                topology.groups.end(),
                                [&binding](const auto &group)
                                {
                                    return group.group_id == binding.group_id;
                                });
                            if (group_it == topology.groups.end())
                            {
                                result.error = "transport group has no frozen topology";
                                protocol.fail();
                                return;
                            }
                            std::vector<
                                MoEOverlayDeviceRuntimePublicationFixture *>
                                group_fixtures;
                            const std::optional<std::uint32_t>
                                omitted_ordinal =
                                    options.fault ==
                                                RuntimeCandidateApplyFault::
                                                    OmitFirstDestinationArrival &&
                                            !acquired.batch.entries.empty()
                                        ? std::optional<std::uint32_t>{
                                              acquired.batch.entries.front()
                                                  .ordinal}
                                        : std::nullopt;
                            for (const int participant :
                                 group_it->participant_ids)
                            {
                                auto *const fixture = fixture_for(participant);
                                const bool owns_omitted_arrival =
                                    omitted_ordinal.has_value() &&
                                    acquired.batch.entries[
                                        *omitted_ordinal]
                                            .destination_participant ==
                                        static_cast<std::uint32_t>(
                                            participant);
                                if (!fixture ||
                                    !fixture->enqueuePreparedArrivals(
                                        acquired.batch,
                                        &result.error,
                                        owns_omitted_arrival
                                            ? omitted_ordinal
                                            : std::nullopt))
                                {
                                    protocol.fail();
                                    return;
                                }
                                group_fixtures.push_back(fixture);
                            }
                            // Release only after every event in this group has
                            // been recorded. The maintenance scheduler may now
                            // install exact event dependencies without waiting
                            // for the transfers themselves to finish.
                            std::atomic_ref<bool>(result.transfers_submitted)
                                .store(true, std::memory_order_release);
                            for (auto *fixture : group_fixtures)
                            {
                                if (!fixture->awaitPreparedArrivals(
                                        &result.error))
                                {
                                    protocol.fail();
                                    return;
                                }
                            }
                            if (!protocol.publishPrepared(
                                    acquired.batch, &result.error))
                            {
                                return;
                            }
                            result.prepared = true;
                            if (!options.complete_epoch)
                                return;

                            // The transport worker owns bytes, never policy.
                            // It may commit those bytes only after every GPU in
                            // this group release-publishes its local selector.
                            const auto wait_until = [&](auto predicate)
                            {
                                // Each physical lifecycle edge owns its own
                                // protocol budget. Reusing the command-acquire
                                // deadline would silently shorten a deliberate
                                // reader grace period in this adversarial rig.
                                const auto phase_deadline =
                                    std::chrono::steady_clock::now() +
                                    std::chrono::seconds(10);
                                while (std::chrono::steady_clock::now() <
                                       phase_deadline)
                                {
                                    if (predicate())
                                        return true;
                                    if (protocol.authorityRejected(
                                            acquired.batch))
                                    {
                                        return false;
                                    }
                                    std::this_thread::sleep_for(
                                        std::chrono::microseconds(50));
                                }
                                return false;
                            };
                            if (!wait_until(
                                    [&]
                                    {
                                        return protocol.publicationReady(
                                            acquired.batch);
                                    }))
                            {
                                if (protocol.authorityRejected(
                                        acquired.batch))
                                {
                                    return;
                                }
                                result.error =
                                    "timed out waiting for every participant RCU publication";
                                protocol.fail();
                                return;
                            }
                            if (!protocol.publishPublished(
                                    acquired.batch, &result.error))
                            {
                                return;
                            }
                            result.published = true;

                            // Physical slots from the prior epoch remain live
                            // until every local participant proves its reader
                            // grace period complete. The fixture has no real
                            // payload allocator to reclaim, so this edge is the
                            // exact safe point represented by the production
                            // asynchronous free/event operation.
                            if (!wait_until(
                                    [&]
                                    {
                                        return protocol.retirementRequested(
                                            acquired.batch);
                                    }))
                            {
                                if (protocol.authorityRejected(
                                        acquired.batch))
                                {
                                    return;
                                }
                                result.error =
                                    "timed out waiting for every participant RCU retirement";
                                protocol.fail();
                                return;
                            }
                            if (!protocol.publishRetired(
                                    acquired.batch, &result.error))
                            {
                                return;
                            }
                            result.retired = true;
                        });
                }
            }

            // Transport workers are live before any participant can enter its
            // physical-readiness wait. Each apply remains ordered behind that
            // participant's pack/reserve work on the same exact stream.
            const auto note_submission = [&submitted](bool accepted)
            {
                // Keep submitting independent producer lanes after one backend
                // rejects a launch so any already-live wait kernel can observe
                // either its producer or the transport failure publication.
                submitted = accepted && submitted;
            };

            // Applying pointer-bearing descriptors before their exact DMA
            // event is in the participant stream DAG is invalid. This bounded
            // scheduler wait observes submission only; copies continue in the
            // background and inference remains entirely independent.
            bool all_transfers_submitted = false;
            const auto submission_deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(5);
            while (submitted &&
                   std::chrono::steady_clock::now() < submission_deadline)
            {
                all_transfers_submitted = std::all_of(
                    transport_results.begin(),
                    transport_results.end(),
                    [](TransportResult &result)
                    {
                        return std::atomic_ref<bool>(
                                   result.transfers_submitted)
                            .load(std::memory_order_acquire);
                    });
                if (all_transfers_submitted)
                    break;
                std::this_thread::yield();
            }
            submitted = submitted && all_transfers_submitted;
            const bool observed_no_movement =
                all_transfers_submitted &&
                std::none_of(
                    transport_results.begin(),
                    transport_results.end(),
                    [](TransportResult &result)
                    {
                        return std::atomic_ref<bool>(result.moves_weights)
                            .load(std::memory_order_acquire);
                    });
            submitted = submitted &&
                (observed_no_movement == options.expect_no_movement);
            if (all_transfers_submitted && !observed_no_movement)
            {
                for (auto &endpoint : endpoints)
                {
                    std::string dependency_error;
                    const bool accepted =
                        endpoint.fixture->enqueuePreparedArrivalDependency(
                            endpoint.stream, &dependency_error);
                    note_submission(accepted);
                    if (!accepted && error && error->empty())
                        *error = std::move(dependency_error);
                }
            }
            if (submitted && !observed_no_movement)
            {
                for (auto &endpoint : endpoints)
                {
                    note_submission(enqueue_action(
                        endpoint,
                        MoEOverlayDeviceControllerAction::
                            ApplyRuntimeCandidate));
                }
            }

            if (options.exercise_reader_grace_period && submitted &&
                !observed_no_movement)
            {
                // Model the request that starts after observation but before
                // the background transfer wave publishes. Its ticket must
                // remain on the base bank throughout the selector flip.
                for (auto &endpoint : endpoints)
                {
                    note_submission(enqueue_reader_acquire(endpoint, 0u));
                    note_submission(endpoint.backend->recordEvent(
                        endpoint.reader_event,
                        endpoint.binding.device.ordinal,
                        endpoint.reader_stream));
                }
                const bool old_reader_wave_complete =
                    await_reader_wave("old-reader acquisition");
                submitted = old_reader_wave_complete && submitted;
                for (auto &endpoint : endpoints)
                {
                    DeviceMoEOverlayEpochStatus status{};
                    DeviceMoEOverlayEpochControl control{};
                    const bool acquired =
                        endpoint.fixture->copyRequestStatus(0u, &status) &&
                        endpoint.fixture->copyEpochControl(&control) &&
                        status.operation == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochOperation::Acquire) &&
                        status.code == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochStatusCode::Success) &&
                        status.epoch == initial_epoch &&
                        control.bank_readers[0] == 1u &&
                        control.bank_readers[1] == 0u;
                    if (!acquired)
                    {
                        append_error(
                            "old reader was not retained on participant " +
                            std::to_string(endpoint.binding.participant_id) +
                            " operation=" +
                            std::to_string(status.operation) + " code=" +
                            std::to_string(status.code) + " epoch=" +
                            std::to_string(status.epoch) + " bank0_readers=" +
                            std::to_string(control.bank_readers[0]) +
                            " bank1_readers=" +
                            std::to_string(control.bank_readers[1]));
                    }
                    submitted = submitted && acquired;
                    evidence->old_epoch_readers_acquired += acquired ? 1u : 0u;
                }
            }

            if (options.complete_epoch && submitted && !observed_no_movement)
            {
                // Fan in complete inactive-bank construction at each group
                // root, then let the sole authority open topology-wide commit.
                for (auto &endpoint : endpoints)
                {
                    if (endpoint.binding.group_root)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AcknowledgePrepared));
                    }
                }
                note_submission(enqueue_action(
                    *leader,
                    MoEOverlayDeviceControllerAction::BeginCommit));

                // Every participant joins the mapped commit edge before its
                // local RCU state is mutated. Submit each phase topology-wide
                // so no driver's queue backpressure can strand a peer producer.
                if (!observed_no_movement)
                {
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AwaitRuntimeCommit));
                    }
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeCandidate));
                    }
                }

                // Host transports can now observe participant publication and
                // release their narrow group record. Group roots acquire that
                // record before the authority makes the new epoch admissible.
                for (auto &endpoint : endpoints)
                {
                    if (endpoint.binding.group_root)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AcknowledgePublished));
                    }
                }
                note_submission(enqueue_action(
                    *leader,
                    MoEOverlayDeviceControllerAction::PublishAdmission));
                note_submission(enqueue_action(
                    *leader,
                    MoEOverlayDeviceControllerAction::
                        BeginDynamicRetirement));
                if (options.exercise_reader_grace_period)
                {
                    // This event is a test-only witness that global admission
                    // moved before the independent inference streams run.
                    note_submission(leader->backend->recordEvent(
                        leader->admission_event,
                        leader->binding.device.ordinal,
                        leader->stream));
                }

                // Retirement readiness is a distinct bounded participant
                // receipt. While the adversarial request holds E, every probe
                // must terminate without publishing readiness or occupying its
                // maintenance stream; reclamation is not submitted yet.
                if (!observed_no_movement)
                {
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AwaitRuntimeRetirement));
                    }
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirementReadiness));
                        if (options.exercise_reader_grace_period)
                        {
                            note_submission(endpoint.backend->recordEvent(
                                endpoint.retirement_event,
                                endpoint.binding.device.ordinal,
                                endpoint.stream));
                        }
                    }
                }

                if (options.exercise_reader_grace_period && submitted &&
                    !observed_no_movement)
                {
                    const bool admission_ready = await(
                        leader->backend,
                        leader->admission_event,
                        leader->binding.device.ordinal,
                        std::chrono::seconds(5));
                    if (!admission_ready)
                    {
                        append_error(
                            "candidate admission event timed out on the authority leader");
                    }
                    submitted = admission_ready && submitted;

                    const bool first_probe_complete =
                        await_retirement_wave("bounded retirement-readiness probe");
                    submitted = first_probe_complete && submitted;
                    for (auto &endpoint : endpoints)
                    {
                        DeviceMoEOverlayEpochControl control{};
                        const auto *record =
                            endpoint.binding.local_participant_record;
                        // The readiness kernel also runs on inference streams.
                        // Its probe must leave maintenance-entry evidence intact
                        // so a stalled publication can be diagnosed reliably.
                        const bool maintenance_entry_preserved =
                            record->observed_action == static_cast<std::uint32_t>(
                                MoEOverlayDeviceControllerAction::AwaitRuntimeRetirement);
                        if (!maintenance_entry_preserved)
                            append_error("readiness probe overwrote maintenance action evidence");
                        const bool deferred =
                            maintenance_entry_preserved &&
                            endpoint.fixture->copyEpochControl(&control) &&
                            control.bank_readers[0] == 1u &&
                            record->status_code == static_cast<std::uint32_t>(
                                MoEOverlayDeviceControllerError::None) &&
                            record->retirement_ready_epoch != initial_epoch &&
                            record->retired_epoch == 0u;
                        submitted = deferred && submitted;
                        evidence->retirement_readiness_deferrals_observed +=
                            deferred ? 1u : 0u;
                        if (!deferred)
                        {
                            append_error(
                                "held reader did not defer retirement readiness for participant " +
                                std::to_string(
                                    endpoint.binding.participant_id) +
                                " readers=" +
                                std::to_string(control.bank_readers[0]) +
                                " readiness_epoch=" +
                                std::to_string(
                                    record->retirement_ready_epoch) +
                                " retired_epoch=" +
                                std::to_string(record->retired_epoch));
                        }
                    }

                    // A second request now acquires and releases the candidate
                    // bank while the old request still pins the retiring bank.
                    // The completed Busy attempt leaves every GPU queue free.
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_reader_acquire(endpoint, 1u));
                        note_submission(enqueue_reader_release(endpoint, 1u));
                        note_submission(endpoint.backend->recordEvent(
                            endpoint.reader_event,
                            endpoint.binding.device.ordinal,
                            endpoint.reader_stream));
                    }
                    const bool new_reader_wave_complete =
                        await_reader_wave("new-reader completion");
                    submitted = new_reader_wave_complete && submitted;

                    // Releasing the original ticket is the only edge that may
                    // unblock old-bank reclamation. No maintenance or inference
                    // stream is synchronized to create that ordering.
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_reader_release(endpoint, 0u));
                        note_submission(endpoint.backend->recordEvent(
                            endpoint.reader_event,
                            endpoint.binding.device.ordinal,
                            endpoint.reader_stream));
                    }
                    const bool release_wave_complete =
                        await_reader_wave("old-reader release");
                    submitted = release_wave_complete && submitted;

                    /* The release epilogue authors one immutable readiness
                     * word per participant. This rig submits the same bounded
                     * action after the witnessed release, then verifies every
                     * topology receipt before submitting reclamation exactly
                     * once. The scheduler chooses neither bank nor epoch. */
                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirementReadiness));
                        note_submission(endpoint.backend->recordEvent(
                            endpoint.retirement_event,
                            endpoint.binding.device.ordinal,
                            endpoint.stream));
                    }
                    const bool readiness_complete = await_retirement_wave(
                        "topology retirement-readiness publication");
                    const bool all_readers_ready = readiness_complete &&
                        std::all_of(
                            endpoints.begin(),
                            endpoints.end(),
                            [initial_epoch](const Endpoint &endpoint)
                            {
                                return endpoint.binding
                                           .local_participant_record
                                           ->retirement_ready_epoch ==
                                    initial_epoch;
                            });
                    submitted = submitted && all_readers_ready;

                    /* Begin one synthetic E+1 admission after every E
                     * readiness receipt. The topology proof makes this guard
                     * unrelated to the retiring bank, so reclamation must not
                     * wait for or reject it. Keeping the guard nonzero through
                     * the retirement graph deterministically covers the race
                     * that live concurrent inference previously exposed. */
                    bool post_readiness_guards_installed = all_readers_ready;
                    for (auto &endpoint : endpoints)
                    {
                        const bool installed = all_readers_ready &&
                            endpoint.fixture
                                ->setPostReadinessAcquisitionGuardForTest(1u);
                        post_readiness_guards_installed =
                            post_readiness_guards_installed && installed;
                        evidence->post_readiness_acquisition_guards +=
                            installed ? 1u : 0u;
                        if (!installed)
                        {
                            append_error(
                                "could not install post-readiness E+1 acquisition guard for participant " +
                                std::to_string(
                                    endpoint.binding.participant_id));
                        }
                    }
                    submitted =
                        submitted && post_readiness_guards_installed;

                    for (auto &endpoint : endpoints)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirement));
                        note_submission(endpoint.backend->recordEvent(
                            endpoint.retirement_event,
                            endpoint.binding.device.ordinal,
                            endpoint.stream));
                    }
                    const bool retirement_complete = await_retirement_wave(
                        "single topology-ready retirement epoch");

                    /* Restore the test-owned guard after the exact retirement
                     * events. No inference or maintenance stream is joined. */
                    bool post_readiness_guards_cleared = true;
                    for (auto &endpoint : endpoints)
                    {
                        post_readiness_guards_cleared =
                            endpoint.fixture
                                ->setPostReadinessAcquisitionGuardForTest(0u) &&
                            post_readiness_guards_cleared;
                    }
                    const bool all_retired = retirement_complete &&
                        post_readiness_guards_cleared &&
                        std::all_of(
                            endpoints.begin(),
                            endpoints.end(),
                            [initial_epoch](const Endpoint &endpoint)
                            {
                                return endpoint.binding
                                           .local_participant_record
                                           ->retired_epoch == initial_epoch;
                            });
                    submitted = submitted && all_retired;
                }

                if (!options.exercise_reader_grace_period && submitted &&
                    !observed_no_movement)
                {
                    /* Match the retained production Publish -> Retire edge.
                     * The mapped readiness words are scheduler receipts only;
                     * they carry no placement state. Once every participant
                     * has proved the old bank reader-free, submit all local
                     * retirement graphs as one independent wavefront. */
                    const auto readiness_deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::seconds(5);
                    bool topology_ready = false;
                    while (std::chrono::steady_clock::now() <
                           readiness_deadline)
                    {
                        topology_ready = std::all_of(
                            endpoints.begin(),
                            endpoints.end(),
                            [initial_epoch](const Endpoint &endpoint)
                            {
                                return endpoint.binding
                                           .local_participant_record
                                           ->retirement_ready_epoch ==
                                    initial_epoch;
                            });
                        if (topology_ready)
                            break;
                        const auto controller_state =
                            std::atomic_ref<const std::uint32_t>(
                                transport_bindings.front()
                                    .controller->state)
                                .load(std::memory_order_acquire);
                        if (controller_state ==
                            static_cast<std::uint32_t>(
                                MoEOverlayDeviceControllerState::Error))
                        {
                            break;
                        }
                        std::this_thread::yield();
                    }
                    const bool authority_rejected =
                        std::atomic_ref<const std::uint32_t>(
                            transport_bindings.front().controller->state)
                            .load(std::memory_order_acquire) ==
                        static_cast<std::uint32_t>(
                            MoEOverlayDeviceControllerState::Error);
                    submitted = submitted &&
                        (topology_ready || authority_rejected);
                    if (topology_ready)
                    {
                        for (auto &endpoint : endpoints)
                        {
                            note_submission(enqueue_action(
                                endpoint,
                                MoEOverlayDeviceControllerAction::
                                    PublishRuntimeRetirement));
                        }
                    }
                }

                // Only a successful device-owned retirement lets the physical
                // workers publish their completion words and release these
                // group/leader fan-in actions.
                for (auto &endpoint : endpoints)
                {
                    if (endpoint.binding.group_root)
                    {
                        note_submission(enqueue_action(
                            endpoint,
                            MoEOverlayDeviceControllerAction::
                                AcknowledgeRetired));
                    }
                }
                note_submission(enqueue_action(
                    *leader,
                    MoEOverlayDeviceControllerAction::
                        CompleteDynamicRetirement));
                for (auto &endpoint : endpoints)
                {
                    note_submission(enqueue_action(
                        endpoint,
                        MoEOverlayDeviceControllerAction::
                            AwaitTransactionComplete));
                }
            }

            if (options.complete_epoch && submitted && observed_no_movement)
            {
                /* The retained production author graph closes an empty
                 * durable decision immediately. Every non-authority graph
                 * consumes that exact terminal word; no synthetic
                 * prepare/commit/retire lifecycle is manufactured. */
                for (auto &endpoint : endpoints)
                {
                    note_submission(enqueue_action(
                        endpoint,
                        MoEOverlayDeviceControllerAction::
                            AwaitTransactionComplete));
                }
            }

            for (auto &endpoint : endpoints)
            {
                note_submission(endpoint.backend->recordEvent(
                    endpoint.terminal,
                    endpoint.binding.device.ordinal,
                    endpoint.stream));
            }

            // Poll all participants against one deadline. Sequential waits can
            // multiply a single broken edge by the participant count and then
            // block teardown while a wait kernel is still live.
            std::vector<bool> endpoint_complete(endpoints.size(), false);
            bool all_endpoints_complete = false;
            const auto endpoint_deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(10);
            while (submitted &&
                   std::chrono::steady_clock::now() < endpoint_deadline)
            {
                all_endpoints_complete = true;
                for (std::size_t index = 0u;
                     index < endpoints.size();
                     ++index)
                {
                    if (!endpoint_complete[index])
                    {
                        bool ready = false;
                        if (!endpoints[index].backend->queryEvent(
                                endpoints[index].terminal,
                                endpoints[index].binding.device.ordinal,
                                &ready))
                        {
                            submitted = false;
                            break;
                        }
                        endpoint_complete[index] = ready;
                    }
                    all_endpoints_complete =
                        all_endpoints_complete && endpoint_complete[index];
                }
                if (all_endpoints_complete)
                    break;
                std::this_thread::yield();
            }

            bool complete = submitted && all_endpoints_complete;
            if (!complete)
            {
                // Publish a physical failure on every host-owned lane so an
                // ApplyRuntimeCandidate wait exits before stream destruction.
                // The diagnostic reads only the protocol's sanctioned const
                // controller view; it never authors placement or an epoch.
                if (error && !transport_bindings.empty())
                {
                    const auto *const controller =
                        transport_bindings.front().controller;
                    std::string timeout_error =
                        "runtime candidate endpoint timeout: state=" +
                        std::to_string(controller->state) +
                        ", error=" +
                        std::to_string(controller->error_code) +
                        ", transaction=" +
                        std::to_string(controller->transaction_id) +
                        ", command_transaction=" +
                        std::to_string(controller->command_transaction) +
                        ", command_count=" +
                        std::to_string(
                            transport_bindings.front().command->command_count) +
                        ", projected_gain_ns=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->projected_service_gain_ns) +
                        ", projected_transfer_ns=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->projected_transfer_and_repack_ns) +
                        ", projected_interference_ns=" +
                        std::to_string(
                            transport_bindings.front()
                                .command
                                ->projected_inference_interference_ns) +
                        ", projected_net_ns=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->projected_net_benefit_ns) +
                        ", accepted_cycles=" +
                        std::to_string(
                            transport_bindings.front().command->accepted_cycles) +
                        ", rejected_cycles=" +
                        std::to_string(
                            transport_bindings.front().command->rejected_cycles) +
                        ", payoff_rejected_cycles=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->payoff_rejected_cycles) +
                        ", residency_rejected_cycles=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->residency_rejected_cycles) +
                        ", priority_before=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->priority_cost_before) +
                        ", priority_after=" +
                        std::to_string(
                            transport_bindings.front()
                                .command->priority_cost_after);
                    for (std::size_t index = 0u;
                         index < endpoints.size();
                         ++index)
                    {
                        if (!endpoint_complete[index])
                        {
                            timeout_error += ", pending_participant=" +
                                std::to_string(
                                    endpoints[index].binding.participant_id);
                        }
                    }
                    for (const auto &binding : transport_bindings)
                    {
                        timeout_error += ", group=" +
                            std::to_string(binding.group_id) +
                            " transport_state=" +
                            std::to_string(binding.transport->state) +
                            " prepared=" +
                            std::to_string(
                                binding.transport->prepared_transaction) +
                            " transport_error=" +
                            std::to_string(binding.transport->status_code);
                    }
                    if (!error->empty())
                        *error += "; ";
                    *error += std::move(timeout_error);
                }
                for (const auto &binding : transport_bindings)
                {
                    MoEOverlayDeviceTransportProtocol protocol(binding);
                    protocol.fail();
                }

                const auto unwind_deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(5);
                while (std::chrono::steady_clock::now() < unwind_deadline)
                {
                    bool unwound = true;
                    for (std::size_t index = 0u;
                         index < endpoints.size();
                         ++index)
                    {
                        if (!endpoint_complete[index])
                        {
                            bool ready = false;
                            if (endpoints[index].backend->queryEvent(
                                    endpoints[index].terminal,
                                    endpoints[index].binding.device.ordinal,
                                    &ready))
                            {
                                endpoint_complete[index] = ready;
                            }
                        }
                        unwound = unwound && endpoint_complete[index];
                    }
                    if (unwound)
                        break;
                    std::this_thread::yield();
                }
            }
            for (auto &worker : workers)
                worker.join();

            if (options.exercise_reader_grace_period)
            {
                // Semantic inspection is terminal-only. At this point every
                // maintenance and inference event is complete, so evidence
                // collection cannot perturb the overlap being certified.
                for (auto &endpoint : endpoints)
                {
                    DeviceMoEOverlayEpochStatus old_status{};
                    DeviceMoEOverlayEpochStatus new_status{};
                    const bool old_released =
                        endpoint.fixture->copyRequestStatus(
                            0u, &old_status) &&
                        old_status.operation == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochOperation::Release) &&
                        old_status.code == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochStatusCode::Success) &&
                        old_status.epoch == initial_epoch;
                    const bool new_completed =
                        endpoint.fixture->copyRequestStatus(
                            1u, &new_status) &&
                        new_status.operation == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochOperation::Release) &&
                        new_status.code == static_cast<std::uint32_t>(
                            DeviceMoEOverlayEpochStatusCode::Success) &&
                        new_status.epoch == initial_epoch + 1u;
                    complete = complete && old_released && new_completed;
                    evidence->old_epoch_readers_released +=
                        old_released ? 1u : 0u;
                    evidence->new_epoch_readers_completed +=
                        new_completed ? 1u : 0u;
                }
            }
            for (const auto &result : transport_results)
            {
                evidence->transport_groups_prepared +=
                    result.prepared ? 1u : 0u;
                evidence->transport_groups_published +=
                    result.published ? 1u : 0u;
                evidence->transport_groups_retired +=
                    result.retired ? 1u : 0u;
                if (!result.error.empty())
                {
                    complete = false;
                    if (error)
                    {
                        if (!error->empty())
                            *error += "; ";
                        *error += result.error;
                    }
                }
            }
            if (!complete)
            {
                if (error && error->empty())
                {
                    *error = submitted
                                 ? "runtime candidate apply did not reach every terminal event"
                                 : "runtime candidate apply submission was rejected";
                }
                if (error && !transport_bindings.empty())
                {
                    const auto *const controller =
                        transport_bindings.front().controller;
                    *error += "; controller_state=" +
                        std::to_string(controller->state) +
                        ", controller_error=" +
                        std::to_string(controller->error_code);
                    for (const auto &binding : transport_bindings)
                    {
                        for (std::uint32_t index = 0u;
                             index < binding.participant_record_count;
                             ++index)
                        {
                            const auto &record =
                                binding.participant_records[index];
                            *error += ", participant=" +
                                std::to_string(record.participant_id) +
                                " status=" +
                                std::to_string(record.status_code) +
                                " prepared=" +
                                std::to_string(
                                    record.prepared_transaction) +
                                " published=" +
                                std::to_string(
                                    record.published_transaction) +
                                " retirement_ready=" +
                                std::to_string(
                                    record.retirement_ready_epoch) +
                                " retired=" +
                                std::to_string(record.retired_epoch);
                        }
                    }
                    for (const auto &fixture : fixtures)
                    {
                        MoEOverlayDeviceRuntimeApplyStatus apply{};
                        std::vector<DeviceMoELayerRuntime> layers;
                        DeviceMoEOverlayEpochControl control{};
                        DeviceMoEOverlayEpochStatus status{};
                        if (fixture && fixture->copyEvidence(
                                           &apply,
                                           &layers,
                                           &control,
                                           &status))
                        {
                            *error += ", runtime_participant=" +
                                std::to_string(fixture->participantId()) +
                                " apply_code=" +
                                std::to_string(apply.code) +
                                " apply_transaction=" +
                                std::to_string(apply.transaction_id) +
                                " apply_bank=" +
                                std::to_string(apply.candidate_bank) +
                                " epoch_operation=" +
                                std::to_string(status.operation) +
                                " epoch_code=" +
                                std::to_string(status.code) +
                                " epoch_bank=" +
                                std::to_string(status.bank) +
                                " bank0_state=" +
                                std::to_string(control.bank_states[0]) +
                                " bank1_state=" +
                                std::to_string(control.bank_states[1]) +
                                " bank0_readers=" +
                                std::to_string(control.bank_readers[0]) +
                                " bank1_readers=" +
                                std::to_string(control.bank_readers[1]) +
                                " acquisitions=" +
                                std::to_string(
                                    control.acquisitions_in_flight) +
                                " selector=" +
                                std::to_string(control.published_selector);
                        }
                    }
                }
                cleanup();
                return false;
            }

            bool copied = leader->backend->deviceToHost(
                              &evidence->controller,
                              leader->binding.controller,
                              sizeof(evidence->controller),
                              leader->binding.device.ordinal,
                              leader->stream) &&
                leader->backend->deviceToHost(
                    &evidence->command,
                    leader->binding.command,
                    sizeof(evidence->command),
                    leader->binding.device.ordinal,
                    leader->stream) &&
                leader->backend->deviceToHost(
                    &evidence->policy,
                    policy_device,
                    sizeof(evidence->policy),
                    leader->binding.device.ordinal,
                    leader->stream);
            evidence->commands.resize(evidence->command.command_count);
            if (copied && !evidence->commands.empty())
            {
                copied = leader->backend->deviceToHost(
                    evidence->commands.data(),
                    leader->binding.command_entries,
                    evidence->commands.size() *
                        sizeof(MoEOverlayDeviceMovementCommand),
                    leader->binding.device.ordinal,
                    leader->stream);
            }
            evidence->economy_last_moved.resize(
                static_cast<std::size_t>(leader->binding.layout->num_layers) *
                leader->binding.layout->num_experts);
            if (copied)
            {
                copied = leader->backend->deviceToHost(
                    evidence->economy_last_moved.data(),
                    leader->binding.economy_last_moved,
                    evidence->economy_last_moved.size() *
                        sizeof(std::uint64_t),
                    leader->binding.device.ordinal,
                    leader->stream);
            }
            evidence->apply_statuses.resize(fixtures.size());
            evidence->runtime_layers.resize(fixtures.size());
            evidence->epoch_controls.resize(fixtures.size());
            evidence->epoch_statuses.resize(fixtures.size());
            for (std::size_t index = 0u;
                 copied && index < fixtures.size();
                 ++index)
            {
                copied = fixtures[index]->copyEvidence(
                    &evidence->apply_statuses[index],
                    &evidence->runtime_layers[index],
                    &evidence->epoch_controls[index],
                    &evidence->epoch_statuses[index]);
            }
            for (const auto &binding : transport_bindings)
            {
                if (!copied)
                    break;
                evidence->participant_records.insert(
                    evidence->participant_records.end(),
                    binding.participant_records,
                    binding.participant_records +
                        binding.participant_record_count);
            }
            if (!copied && error)
                *error = "terminal runtime candidate evidence copy failed";
            cleanup();
            return copied;
        }

        /** Both synthetic rank-local views of one production controller mapping. */
        struct ControllerFabricPair
        {
            std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
            std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        };

        /**
         * @brief Construct both rank views concurrently so first-touch precedes registration.
         * @param resolved_topology Immutable production-shaped device topology.
         * @param policy_input Exact mapped snapshot and command geometry.
         * @return Fully registered CUDA/ROCm fabric pair.
         */
        ControllerFabricPair makeControllerFabricPair(
            std::shared_ptr<const MoEOverlayDeviceControllerTopology>
                resolved_topology,
            const MoEOverlayDevicePlacementPolicyInput &policy_input)
        {
            ControllerFabricPair result;
            std::exception_ptr cuda_error;
            std::exception_ptr rocm_error;
            const auto config = [&](int rank)
            {
                return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                    .mpi_ctx = context(rank),
                    .topology = resolved_topology,
                    .num_layers = policy_input.num_layers,
                    .num_experts = policy_input.num_experts,
                    .command_capacity = policy_input.command_capacity,
                    .initial_durable_epoch = policy_input.base_epoch,
                    .payload_bytes_per_layer =
                        policy_input.payload_bytes_per_layer,
                    .initial_owner_participants = initialOwners(policy_input),
                    .minimum_window_activations =
                        policy_input.minimum_window_activations,
                    .maximum_cycles_per_wave =
                        policy_input.maximum_cycles_per_wave,
                    .dynamic_imbalance_threshold_per_mille =
                        policy_input.dynamic_imbalance_threshold_per_mille,
                    .dynamic_minimum_improvement_per_mille =
                        policy_input.dynamic_minimum_improvement_per_mille,
                    .dynamic_maximum_cycles_per_layer =
                        policy_input.dynamic_maximum_cycles_per_layer,
                    .dynamic_maximum_commands_per_wave =
                        policy_input.dynamic_maximum_commands_per_wave,
                };
            };
            std::thread cuda_builder(
                [&]
                {
                    try
                    {
                        result.cuda_rank = std::make_unique<
                            MoEOverlayNodeLocalDeviceControllerFabric>(
                            config(0));
                    }
                    catch (...)
                    {
                        cuda_error = std::current_exception();
                    }
                });
            std::thread rocm_builder(
                [&]
                {
                    try
                    {
                        result.rocm_rank = std::make_unique<
                            MoEOverlayNodeLocalDeviceControllerFabric>(
                            config(1));
                    }
                    catch (...)
                    {
                        rocm_error = std::current_exception();
                    }
                });
            cuda_builder.join();
            rocm_builder.join();
            if (cuda_error)
                std::rethrow_exception(cuda_error);
            if (rocm_error)
                std::rethrow_exception(rocm_error);
            if (!result.cuda_rank || !result.rocm_rank)
            {
                throw std::runtime_error(
                    "controller fabric pair construction returned an empty rank view");
            }
            return result;
        }

        /**
         * @brief One real GPU endpoint for the symmetric epoch admission proof.
         *
         * The endpoint owns the same arena and public MoE kernel used by captured
         * production boundaries. Setup and terminal evidence may wait; acquire,
         * release, and admission publication themselves are exact-stream async.
         */
        class InferenceEpochBarrierEndpoint final
        {
        public:
            /** Allocate one model-lifetime arena against its mapped fabric alias. */
            InferenceEpochBarrierEndpoint(
                IBackend *backend,
                MoEOverlayDeviceControllerParticipantBinding binding,
                std::uint64_t initial_epoch)
                : backend_(backend), binding_(std::move(binding))
            {
                if (!backend_ || !binding_.valid() ||
                    !binding_.inference_epoch_member ||
                    !binding_.inference_epoch_record || initial_epoch == 0u)
                {
                    throw std::invalid_argument(
                        "inference epoch endpoint requires one continuation GPU binding");
                }
                arena_ = std::make_shared<DeviceMoEOverlayEpochArena>(
                    DeviceMoEOverlayEpochArena::Config{
                        .device_id = binding_.device,
                        .initial_epoch = initial_epoch,
                        .initial_bank = 0u,
                        .request_slot_capacity = 1u,
                        .external_admission_epoch =
                            &binding_.controller->admission_epoch,
                        .external_admission_lifetime = binding_.lifetime,
                        .admission_barrier = {
                            .record = binding_.inference_epoch_record,
                            .participant_id = static_cast<std::uint32_t>(
                                binding_.participant_id),
                        },
                    });
                kernel_ = llaminar::v2::kernels::KernelFactory::createMoEKernel(
                    binding_.device);
                try
                {
                    inference_stream_ =
                        backend_->createStream(binding_.device.ordinal);
                    maintenance_stream_ =
                        backend_->createStream(binding_.device.ordinal);
                    inference_event_ =
                        backend_->createEvent(binding_.device.ordinal);
                    maintenance_event_ =
                        backend_->createEvent(binding_.device.ordinal);
                    if (!arena_ || !kernel_ || !inference_stream_ ||
                        !maintenance_stream_ || !inference_event_ ||
                        !maintenance_event_)
                    {
                        throw std::runtime_error(
                            "inference epoch endpoint could not allocate exact streams/events");
                    }
                }
                catch (...)
                {
                    releaseResources(/*drain=*/false);
                    throw;
                }
            }

            /** Drain test-owned streams, then release events and model storage. */
            ~InferenceEpochBarrierEndpoint()
            {
                releaseResources(/*drain=*/true);
            }

            InferenceEpochBarrierEndpoint(
                const InferenceEpochBarrierEndpoint &) = delete;
            InferenceEpochBarrierEndpoint &operator=(
                const InferenceEpochBarrierEndpoint &) = delete;

            /** Publish the arena's preallocated next local epoch on maintenance. */
            [[nodiscard]] bool publishNextLocalEpoch()
            {
                const MoEKernelLaunchContext launch{
                    .stream = maintenance_stream_,
                    .workspace = nullptr,
                };
                return kernel_->reserveMoEOverlayEpochCandidate(
                           launch,
                           arena_->control(),
                           arena_->maintenanceEpoch(),
                           arena_->maintenanceStatus()) &&
                       kernel_->markMoEOverlayEpochCandidateReady(
                           launch,
                           arena_->control(),
                           arena_->maintenanceEpoch(),
                           arena_->maintenanceStatus()) &&
                       kernel_->publishMoEOverlayEpochCandidate(
                           launch,
                           arena_->control(),
                           arena_->maintenanceEpoch(),
                           arena_->maintenanceStatus()) &&
                       recordMaintenanceTerminal() &&
                       await(
                           backend_,
                           maintenance_event_,
                           binding_.device.ordinal,
                           std::chrono::seconds(5));
            }

            /** Enqueue one transaction-wide guarded acquire without host waiting. */
            [[nodiscard]] bool enqueueAcquire()
            {
                return kernel_->acquireMoEOverlayEpoch(
                           inferenceLaunch(),
                           arena_->control(),
                           arena_->requestTicket(0u),
                           arena_->requestStatus(0u),
                           arena_->externalAdmissionEpoch(),
                           arena_->admissionBarrier()) &&
                       backend_->recordEvent(
                           inference_event_,
                           binding_.device.ordinal,
                           inference_stream_);
            }

            /** Enqueue release of the exact ticket selected by @ref enqueueAcquire. */
            [[nodiscard]] bool enqueueRelease()
            {
                return kernel_->releaseMoEOverlayEpoch(
                           inferenceLaunch(),
                           arena_->control(),
                           arena_->requestTicket(0u),
                           arena_->requestStatus(0u)) &&
                       backend_->recordEvent(
                           inference_event_,
                           binding_.device.ordinal,
                           inference_stream_);
            }

            /** @return Whether the latest inference boundary reached its event. */
            [[nodiscard]] bool awaitInference() const
            {
                return await(
                    backend_,
                    inference_event_,
                    binding_.device.ordinal,
                    std::chrono::seconds(5));
            }

            /** Copy this participant's mapped arrival lane for adversarial ordering. */
            [[nodiscard]] bool copyArrival(std::uint64_t *arrival) const
            {
                return arrival && backend_->deviceToHost(
                    arrival,
                    &binding_.inference_epoch_record->arrival_sequence[
                        binding_.participant_id],
                    sizeof(*arrival),
                    binding_.device.ordinal,
                    maintenance_stream_);
            }

            /** Publish a new topology admission through the authority alias. */
            [[nodiscard]] bool publishGlobalAdmission(std::uint64_t epoch)
            {
                return backend_->hostToDeviceOnStream(
                           &binding_.controller->admission_epoch,
                           &epoch,
                           sizeof(epoch),
                           binding_.device.ordinal,
                           maintenance_stream_) &&
                       recordMaintenanceTerminal() &&
                       await(
                           backend_,
                           maintenance_event_,
                           binding_.device.ordinal,
                           std::chrono::seconds(5));
            }

            /** Copy the completed request ticket and semantic status. */
            [[nodiscard]] bool copyAcquireEvidence(
                DeviceMoEOverlayEpochTicket *ticket,
                DeviceMoEOverlayEpochStatus *status) const
            {
                return ticket && status && backend_->deviceToHost(
                           ticket,
                           arena_->requestTicket(0u),
                           sizeof(*ticket),
                           binding_.device.ordinal,
                           maintenance_stream_) &&
                       backend_->deviceToHost(
                           status,
                           arena_->requestStatus(0u),
                           sizeof(*status),
                           binding_.device.ordinal,
                           maintenance_stream_);
            }

            /** Copy the complete mapped barrier through this endpoint's alias. */
            [[nodiscard]] bool copyBarrier(
                MoEOverlayDeviceControllerInferenceEpochRecord *record) const
            {
                return record && backend_->deviceToHost(
                    record,
                    binding_.inference_epoch_record,
                    sizeof(*record),
                    binding_.device.ordinal,
                    maintenance_stream_);
            }

            /** @return Immutable global participant id. */
            [[nodiscard]] int participantId() const noexcept
            {
                return binding_.participant_id;
            }

        private:
            /** @return Exact stream binding used by public inference kernels. */
            [[nodiscard]] MoEKernelLaunchContext inferenceLaunch() const noexcept
            {
                return {
                    .stream = inference_stream_,
                    .workspace = nullptr,
                };
            }

            /** Record the reusable maintenance terminal after preceding work. */
            [[nodiscard]] bool recordMaintenanceTerminal()
            {
                return backend_->recordEvent(
                    maintenance_event_,
                    binding_.device.ordinal,
                    maintenance_stream_);
            }

            /**
             * @brief Release every partially or fully constructed GPU handle.
             *
             * Construction has no submitted work, while normal destruction
             * drains the two test-owned streams before destroying their reusable
             * terminal events. Keeping this cleanup idempotent makes failure at
             * any allocation boundary leak-free without changing production
             * stream semantics.
             */
            void releaseResources(bool drain) noexcept
            {
                if (!backend_)
                    return;
                if (drain && inference_stream_)
                    (void)backend_->synchronizeStream(
                        inference_stream_, binding_.device.ordinal);
                if (drain && maintenance_stream_)
                    (void)backend_->synchronizeStream(
                        maintenance_stream_, binding_.device.ordinal);
                if (inference_event_)
                    backend_->destroyEvent(
                        inference_event_, binding_.device.ordinal);
                if (maintenance_event_)
                    backend_->destroyEvent(
                        maintenance_event_, binding_.device.ordinal);
                if (inference_stream_)
                    backend_->destroyStream(
                        inference_stream_, binding_.device.ordinal);
                if (maintenance_stream_)
                    backend_->destroyStream(
                        maintenance_stream_, binding_.device.ordinal);
                inference_event_ = nullptr;
                maintenance_event_ = nullptr;
                inference_stream_ = nullptr;
                maintenance_stream_ = nullptr;
            }

            IBackend *backend_ = nullptr;
            MoEOverlayDeviceControllerParticipantBinding binding_;
            std::shared_ptr<DeviceMoEOverlayEpochArena> arena_;
            std::unique_ptr<IMoEKernel> kernel_;
            void *inference_stream_ = nullptr;
            void *maintenance_stream_ = nullptr;
            void *inference_event_ = nullptr;
            void *maintenance_event_ = nullptr;
        };
    } // namespace

    /** Prove immutable mapped staging and controller-certified reuse on one backend. */
    void proveMappedParticipantInbox(DeviceType type)
    {
        IBackend *const backend = type == DeviceType::CUDA
            ? getCUDABackend() : getROCmBackend();
        if (!backend || !getCUDABackend() || !getROCmBackend() ||
            getCUDABackend()->deviceCount() < 2 ||
            getROCmBackend()->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto policy_input = adversarialPolicyInput();
        ASSERT_GE(resolved_topology->participants.size(), 5u);
        ASSERT_EQ(resolved_topology->participants[2].world_rank, 1);
        ASSERT_EQ(resolved_topology->participants[3].world_rank, 1);
        ASSERT_EQ(resolved_topology->participants[4].world_rank, 1);

        const auto participant = type == DeviceType::CUDA ? 0u : 2u;
        auto fabrics = makeControllerFabricPair(resolved_topology, policy_input);
        auto &fabric = type == DeviceType::CUDA ? *fabrics.cuda_rank : *fabrics.rocm_rank;
        const auto transport_binding = fabric.transportBinding(
            resolved_topology->groupForParticipant(participant)->group_id);
        MoEOverlayDeviceTransportProtocol protocol(transport_binding);
        MoEOverlayDeviceRuntimePublicationFixture fixture(
            backend,
            resolved_topology->participants[participant],
            *resolved_topology,
            policy_input);
        ScopedGPUStream consumer_stream(
            resolved_topology->participants[participant].device);
        MoEOverlayDevicePreparedArrivalInbox inbox({
            .backend = backend,
            .runtime_binding = fixture.runtimeBinding(),
            .command_capacity = policy_input.command_capacity,
            .consumer_stream = consumer_stream.get(),
            .perf_device = "sibling-rank-arrival-regression",
        });

        MoEOverlayDeviceTransportCommandBatch command;
        command.participant_count = static_cast<std::uint32_t>(
            resolved_topology->participants.size());
        command.num_layers = policy_input.num_layers;
        command.num_experts = policy_input.num_experts;
        command.entries = {
            MoEOverlayDeviceMovementCommand{
                .op = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementOp::DurableMove),
                .ordinal = 0u,
                .layer = 0u,
                .expert = 0u,
                .source_participant = 4u,
                .destination_participant = 3u,
                .payload_slot = 0u,
                .flags = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                .payload_bytes = 4096u,
                .source_epoch = policy_input.base_epoch,
                .candidate_epoch = policy_input.base_epoch + 1u,
            },
            MoEOverlayDeviceMovementCommand{
                .op = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementOp::DurableMove),
                .ordinal = 1u,
                .layer = 0u,
                .expert = 1u,
                .source_participant = 3u,
                .destination_participant = 4u,
                .payload_slot = 1u,
                .flags = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                .payload_bytes = 4096u,
                .source_epoch = policy_input.base_epoch,
                .candidate_epoch = policy_input.base_epoch + 1u,
            },
        };
        command.header.kind = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement);
        command.header.demand_phase = static_cast<std::uint32_t>(
            MoEOverlayDeviceDemandPhase::Decode);
        command.header.command_count = static_cast<std::uint32_t>(
            command.entries.size());
        command.header.topology_fingerprint =
            resolved_topology->topology_fingerprint;
        command.header.transaction_id = 9u;
        command.header.base_epoch = policy_input.base_epoch;
        command.header.candidate_epoch = policy_input.base_epoch + 1u;
        command.header.command_digest = commandDigest(
            command.entries.data(), command.header.command_count);
        command.header.packed_weight_bytes = 8192u;
        command.header.parallel_command_count = command.header.command_count;
        command.header.movement_round_count = 1u;
        command.header.same_priority_makespan_before = 2u;
        command.header.same_priority_makespan_after = 1u;
        command.header.accepted_cycles = 1u;
        command.header.same_priority_moves = 2u;
        command.header.changed_layers = 1u;
        command.header.projected_service_gain_ns = 1'000u;
        command.header.projected_transfer_and_repack_ns = 100u;
        command.header.projected_inference_interference_ns = 100u;
        command.header.projected_net_benefit_ns = 800u;
        ASSERT_TRUE(command.valid());

        const auto batch = makeMoEOverlayDevicePhysicalMovementBatch(
            command, *resolved_topology);
        ASSERT_TRUE(batch.valid());
        MoEOverlayParticipantPreparedTransfers prepared;
        prepared.status = MoEOverlayResidencyStageStartStatus::Started;
        prepared.migrations.resize(batch.migrations.size());
        for (auto &migration : prepared.migrations)
        {
            /*
             * Both lifetimes belong to sibling ROCm devices on this process.
             * They are deliberately incomplete because this participant must
             * never inspect or export them.
             */
            migration.destination_arrival =
                std::make_shared<MoEOverlayPreparedExpertArrival>();
        }

        std::string error;
        ASSERT_TRUE(inbox.stage(batch, prepared, &error)) << error;
        EXPECT_FALSE(inbox.stage(batch, prepared, &error))
            << "An in-flight publication must be immutable";
        EXPECT_FALSE(inbox.finishWave(protocol, command, &error))
            << "Staging is not controller completion";
        // This infrastructure test injects only a terminal device receipt;
        // captured multi-device wave tests below prove its real GPU producer.
        auto *controller = const_cast<MoEOverlayDeviceControllerSharedHeader *>(
            transport_binding.controller);
        __atomic_store_n(&controller->completed_transaction,
                         command.header.transaction_id - 1u, __ATOMIC_RELEASE);
        EXPECT_FALSE(inbox.finishWave(protocol, command, &error));
        __atomic_store_n(&controller->completed_transaction,
                         command.header.transaction_id, __ATOMIC_RELEASE);
        auto wrong_command = command;
        ++wrong_command.header.command_digest;
        EXPECT_FALSE(inbox.finishWave(protocol, wrong_command, &error));
        EXPECT_TRUE(inbox.finishWave(protocol, command, &error)) << error;
        EXPECT_FALSE(inbox.finishWave(protocol, command, &error));
        EXPECT_FALSE(inbox.stage(batch, prepared, &error))
            << "A retired transaction cannot be staged again";
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CUDAMappedInboxRequiresExactControllerCompletionBeforeReuse)
    {
        proveMappedParticipantInbox(DeviceType::CUDA);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ROCmMappedInboxSkipsSiblingLifetimesAndRequiresExactCompletion)
    {
        proveMappedParticipantInbox(DeviceType::ROCm);
    }

    /**
     * @test A retained empty-completion/open pair cannot erase a slow follower's command.
     *
     * Setup supplies an already-sealed empty command, not a host completion.
     * The real device kernels complete it and open the next phase before either
     * transport worker runs. CUDA and ROCm each own the authority in turn. The
     * same captured graph is replayed twenty times with changing transaction
     * identities; no timing-dependent sleep is needed to force the interleaving.
     */
    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CapturedEmptyCompletionRetainsCommandAcrossNextSnapshotOpen)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 || rocm->deviceCount() < 4)
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";

        for (const DeviceType authority_type : {DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(authority_type == DeviceType::CUDA ? "CUDA authority" : "ROCm authority");
            const auto resolved = topology(authority_type);
            const auto input = adversarialPolicyInput();
            auto fabrics = makeControllerFabricPair(resolved, input);
            auto &authority_fabric = authority_type == DeviceType::CUDA
                                         ? *fabrics.cuda_rank : *fabrics.rocm_rank;
            const auto binding = authority_fabric.participantBinding(
                resolved->leader_participant_id);
            const auto transports = allTransportBindings(*fabrics.cuda_rank, *fabrics.rocm_rank);
            ASSERT_TRUE(binding.authority_leader);
            ASSERT_FALSE(transports.empty());
            IBackend *const backend = authority_type == DeviceType::CUDA ? cuda : rocm;
            const int ordinal = binding.device.gpu_ordinal();
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(binding.device);
            ScopedGPUStream stream(binding.device);
            const auto destroy_event = [backend, ordinal](void *event) {
                if (event) backend->destroyEvent(event, ordinal);
            };
            std::unique_ptr<void, decltype(destroy_event)> terminal(
                backend->createEvent(ordinal), destroy_event);
            std::unique_ptr<IGPUGraphCapture> graph;
            ASSERT_TRUE(kernel && stream.get() && terminal);
            auto &worker = GPUDeviceContextPool::instance().getContext(binding.device);
            bool captured = false;
            worker.submitAndWait([&] {
                graph = worker.createGraphCapture(stream.get());
                if (!graph || !graph->beginCapture()) return;
                const MoEKernelLaunchContext launch{.stream = stream.get()};
                const bool nodes = kernel->runMoEOverlayDeviceControllerAction(launch, {
                    .binding = binding.deviceBinding(),
                    .action = MoEOverlayDeviceControllerAction::CompleteEmptyDynamicDecision,
                }) && kernel->runMoEOverlayDeviceControllerAction(launch, {
                    .binding = binding.deviceBinding(),
                    .action = MoEOverlayDeviceControllerAction::BeginTransaction,
                    .transaction_kind = MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                    .demand_phase = MoEOverlayDeviceDemandPhase::Decode,
                });
                const bool ended = graph->endCapture();
                captured = nodes && ended && graph->instantiate();
            });
            ASSERT_TRUE(captured);

            auto *controller = const_cast<MoEOverlayDeviceControllerSharedHeader *>(transports[0].controller);
            auto *command = const_cast<MoEOverlayDeviceControllerCommandHeader *>(transports[0].command);
            for (std::uint64_t iteration = 0u; iteration < 20u; ++iteration)
            {
                SCOPED_TRACE(iteration);
                const auto transaction = 2u * iteration + 1u;
                // Only fixture admission writes these pages, after the prior
                // replay's terminal. Both completion and next-open are GPU-owned.
                const MoEOverlayDeviceControllerCommandHeader sealed{
                    .kind = static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement),
                    .topology_fingerprint = resolved->topology_fingerprint,
                    .transaction_id = transaction,
                    .base_epoch = input.base_epoch,
                    .candidate_epoch = input.base_epoch,
                    .command_digest = moeOverlayCommandDigestSeed(0u),
                    .demand_phase = static_cast<std::uint32_t>(MoEOverlayDeviceDemandPhase::Prefill),
                };
                *command = sealed;
                controller->transaction_kind = sealed.kind;
                controller->transaction_demand_phase = sealed.demand_phase;
                controller->transaction_id = transaction;
                controller->command_transaction = transaction;
                controller->base_epoch = input.base_epoch;
                controller->candidate_epoch = input.base_epoch;
                std::atomic_ref(controller->state).store(
                    static_cast<std::uint32_t>(MoEOverlayDeviceControllerState::PreparingFollowers),
                    std::memory_order_release);
                ASSERT_TRUE(graph->launchOnStream(stream.get()));
                ASSERT_TRUE(backend->recordEvent(terminal.get(), ordinal, stream.get()));
                ASSERT_TRUE(await(backend, terminal.get(), ordinal, std::chrono::seconds(5)));

                ASSERT_EQ(controller->error_code, 0u);
                EXPECT_EQ(controller->completed_transaction, transaction);
                EXPECT_EQ(controller->transaction_id, transaction + 1u);
                EXPECT_EQ(std::memcmp(command, &sealed, sizeof(sealed)), 0)
                    << "Opening the next transaction mutated a still-readable command";
                for (const auto &transport_binding : transports)
                {
                    MoEOverlayDeviceTransportProtocol transport(transport_binding);
                    const auto acquired = transport.tryAcquire(transaction - 1u);
                    ASSERT_EQ(acquired.status, MoEOverlayDeviceTransportAcquireStatus::Ready)
                        << acquired.error;
                    EXPECT_EQ(acquired.batch.header.transaction_id, transaction);
                    EXPECT_TRUE(transport.transactionComplete(acquired.batch));
                    EXPECT_FALSE(transport.allGroupsSnapshotted(transaction + 1u));
                    std::uint64_t observed = 0u;
                    auto kind = MoEOverlayDeviceControllerTransactionKind::Invalid;
                    auto phase = MoEOverlayDeviceDemandPhase::Invalid;
                    ASSERT_TRUE(transport.snapshotTransactionAfter(transaction, &observed, &kind, &phase));
                    EXPECT_EQ(observed, transaction + 1u);
                    EXPECT_EQ(phase, MoEOverlayDeviceDemandPhase::Decode);
                }
            }
        }
    }

    /** Select movement/observation work independently of full-model geometry. */
    enum class PricedPolicyGeometry
    {
        TwoAxisMovement,
        FullModelObservation,
        FullModelMovement,
    };

    /**
     * Build a full 122B-shaped policy workload without loading model weights.
     *
     * Repeat the adversarial sixteen-expert distribution across all 256 experts
     * and 49 layers. Expensive links reject every cycle for the observation
     * proof; cheap links exercise full-size runtime-bank cloning and publication.
     * All geometry-dependent oracle arrays are rebuilt from the same seed.
     */
    MoEOverlayDevicePlacementPolicyInput fullModelPolicyInput(
        std::uint64_t transfer_cost_ns)
    {
        const auto seed = adversarialPolicyInput();
        auto input = seed;
        input.num_layers = 49u;
        input.num_experts = 256u;
        input.maximum_cycles_per_wave = 2u;
        input.dynamic_maximum_cycles_per_layer = 2u;
        const std::size_t plane = input.num_layers * input.num_experts;
        input.collected_state.assign(input.participants.size() * plane, 0u);
        input.payload_bytes_per_layer.assign(input.num_layers, 4096u);
        auto &economy = *input.economy;
        economy.phase_expert_demand.assign(
            kMoEOverlayDeviceControllerDemandPhaseCount * plane, 0u);
        economy.service_costs.resize(economy.tier_count * input.num_layers *
                                    kMoEOverlayDeviceControllerEconomyServicePhaseCount);
        economy.migration_costs.assign(input.participants.size() *
            input.participants.size() * input.num_layers,
            {.transfer_and_repack_ns = transfer_cost_ns,
             .inference_interference_ns = 0u});
        economy.last_moved_generation.assign(
            plane, kMoEOverlayDeviceControllerNeverMovedGeneration);
        for (std::uint32_t layer = 0u; layer < input.num_layers; ++layer)
        {
            for (std::uint32_t expert = 0u; expert < input.num_experts; ++expert)
            {
                std::uint64_t count = 0u;
                for (std::size_t participant = 0u;
                     participant < input.participants.size(); ++participant)
                {
                    const auto word = seed.collected_state[
                        (participant * seed.num_layers + layer % seed.num_layers) *
                        seed.num_experts + expert % seed.num_experts];
                    input.collected_state[(participant * input.num_layers + layer) *
                                          input.num_experts + expert] = word;
                    count += moe_rebalance_policy::collectedStateActivationCount(word);
                }
                economy.phase_expert_demand[plane + layer * input.num_experts + expert] = count;
            }
            for (std::uint32_t tier = 0u; tier < economy.tier_count; ++tier)
                for (std::uint32_t phase = 0u;
                     phase < kMoEOverlayDeviceControllerEconomyServicePhaseCount; ++phase)
                    economy.service_costs[(tier * input.num_layers + layer) *
                        kMoEOverlayDeviceControllerEconomyServicePhaseCount + phase] =
                        10u + tier * 90u;
        }
        return input;
    }

    /** Prove exact oracle pricing and bounded publication at the selected geometry. */
    void proveTwoAxisWave(
        ExpertHistogramProductionSourceMask economy_sources,
        DeviceType continuation_type = DeviceType::CUDA,
        PricedPolicyGeometry geometry = PricedPolicyGeometry::TwoAxisMovement)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology(continuation_type);
        const bool observation_only = geometry == PricedPolicyGeometry::FullModelObservation;
        const std::uint64_t transfer_cost_ns = observation_only ? 1'000'000'000'000u : 1u;
        auto policy_input = geometry != PricedPolicyGeometry::TwoAxisMovement
            ? fullModelPolicyInput(transfer_cost_ns)
            : boundedTwoAxisPolicyInput();
        for (const auto &participant : resolved_topology->participants)
        {
            auto &metadata = policy_input.participants.at(participant.participant_id);
            metadata.tier_index = participant.tier_idx;
            metadata.tier_priority = resolved_topology->groupForParticipant(
                participant.participant_id)->tier_priority;
        }
        // Deliberately unequal phase prices expose any decode/verifier alias.
        // The CPU oracle retains positive costs for unused phases, but those
        // phases have exactly zero demand and cannot influence its score.
        constexpr std::array<std::uint64_t, 3> phase_prices{1u, 2u, 3u};
        const auto source = economy_sources[0]
            ? moe_runtime_abi::HistogramSource::Decode
            : moe_runtime_abi::HistogramSource::GroupedVerifier;
        auto &economy = policy_input.economy.value();
        const auto plane_words = static_cast<std::size_t>(policy_input.num_layers) *
            policy_input.num_experts;
        for (std::size_t word = 0u; word < plane_words; ++word)
        {
            const auto count = economy.phase_expert_demand[plane_words + word];
            economy.phase_expert_demand[plane_words + word] = 0u;
            economy.phase_expert_demand[static_cast<std::size_t>(source) *
                                           plane_words + word] = count;
        }
        for (std::size_t word = 0u; word < economy.service_costs.size(); ++word)
            economy.service_costs[word] *= phase_prices[word % phase_prices.size()];
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_EQ(expected.evidence.accepted_cycles, observation_only ? 0u : 2u);
        if (!observation_only)
        {
            ASSERT_GT(expected.evidence.promotions, 0u);
            ASSERT_GT(expected.evidence.demotions, 0u);
            if (geometry == PricedPolicyGeometry::TwoAxisMovement)
                ASSERT_GT(expected.evidence.same_priority_moves, 0u);
        }

        auto fabrics = makeControllerFabricPair(
            resolved_topology, policy_input);
        publishSyntheticEconomy(
            *fabrics.cuda_rank,
            *fabrics.rocm_rank,
            *resolved_topology,
            policy_input.num_layers,
            economy_sources,
            phase_prices,
            transfer_cost_ns);
        const auto participant_bindings = allParticipantBindings(
            *fabrics.cuda_rank, *fabrics.rocm_rank);
        const auto transport_bindings = allTransportBindings(
            *fabrics.cuda_rank, *fabrics.rocm_rank);

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &candidate)
                {
                    return candidate.participant_id ==
                           participant.participant_id;
                });
            ASSERT_NE(binding, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding->controller->admission_epoch,
                binding->lifetime,
                source));
        }

        RuntimeCandidateApplyEvidence observed;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &observed,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .expect_no_movement = observation_only})) << error;
        ASSERT_EQ(observed.policy.accepted_cycles, observation_only ? 0u : 2u);
        ASSERT_EQ(observed.participant_records.size(), fixtures.size());
        for (const auto &record : observed.participant_records)
        {
            EXPECT_NE(record.observed_action,
                static_cast<std::uint32_t>(MoEOverlayDeviceControllerAction::Invalid));
            EXPECT_EQ(record.observed_action_transaction,
                      observed.controller.transaction_id)
                << "Every participant must publish device-authored action-entry evidence";
        }
        ASSERT_EQ(observed.commands.size(), expected.commands.size());
        EXPECT_EQ(observed.policy.projected_service_gain_ns,
                  expected.evidence.projected_service_gain_ns);
        EXPECT_EQ(observed.policy.projected_net_benefit_ns,
                  expected.evidence.projected_net_benefit_ns);
        EXPECT_EQ(
            observed.policy.command_digest,
            commandDigest(
                expected.commands.data(),
                static_cast<std::uint32_t>(expected.commands.size())));

        bool advances_tier = false;
        bool advances_participant = false;
        for (std::size_t index = 0u;
             index < observed.commands.size();
             ++index)
        {
            EXPECT_EQ(
                std::memcmp(
                    &observed.commands[index],
                    &expected.commands[index],
                    sizeof(MoEOverlayDeviceMovementCommand)),
                0) << "device/CPU command ABI mismatch at ordinal " << index;
            const auto axis = static_cast<MoEOverlayDeviceMovementAxis>(
                observed.commands[index].flags);
            advances_tier = advances_tier ||
                axis == MoEOverlayDeviceMovementAxis::TierResidency ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
            advances_participant = advances_participant ||
                axis == MoEOverlayDeviceMovementAxis::ParticipantPlacement ||
                axis == MoEOverlayDeviceMovementAxis::Combined;
        }
        EXPECT_EQ(advances_tier, !observation_only);
        // The repeated full-model seed can spend both bounded cycles on tier
        // residency. Exact command bytes still prove the CPU oracle's selected
        // objective; the dedicated two-axis geometry separately requires both.
        if (geometry == PricedPolicyGeometry::TwoAxisMovement || observation_only)
            EXPECT_EQ(advances_participant, !observation_only);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceAuthoredTwoCycleWaveAdvancesBothIndependentPlacementAxes)
    {
        proveTwoAxisWave(kAllExpertHistogramProductionSources);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         MTPPricedTwoAxisWaveDoesNotRequireOrdinaryDecodeCalibration)
    {
        // Serial catch-up is reachable, but only prefill and grouped verifier
        // are recurring costs in the production dynamic-MTP serving regime.
        proveTwoAxisWave({false, true, true});
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ROCmAuthorityPricesMTPVerifierAndPublishesBothPlacementAxes)
    {
        proveTwoAxisWave({false, true, true}, DeviceType::ROCm);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CUDAAuthorityCompletesFullModelUnprofitableMTPObservation)
    {
        proveTwoAxisWave({false, true, true}, DeviceType::CUDA,
                        PricedPolicyGeometry::FullModelObservation);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ROCmAuthorityCompletesFullModelUnprofitableMTPObservation)
    {
        proveTwoAxisWave({false, true, true}, DeviceType::ROCm,
                        PricedPolicyGeometry::FullModelObservation);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CUDAAuthorityPublishesFullModelMTPMovementBank)
    {
        proveTwoAxisWave({false, true, true}, DeviceType::CUDA,
                        PricedPolicyGeometry::FullModelMovement);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ROCmAuthorityPublishesFullModelMTPMovementBank)
    {
        proveTwoAxisWave({false, true, true}, DeviceType::ROCm,
                        PricedPolicyGeometry::FullModelMovement);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceAuthoredDynamicRetirementDoesNotBlockNewCudaOrRocmReaders)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto policy_input = adversarialPolicyInput();
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(expected.hasMovement());
        ASSERT_GT(expected.evidence.promotions, 0u);
        ASSERT_GT(expected.evidence.demotions, 0u);
        ASSERT_GT(expected.evidence.same_priority_moves, 0u);

        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        const auto fabric_config = [&](int rank)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = context(rank),
                .topology = resolved_topology,
                .num_layers = policy_input.num_layers,
                .num_experts = policy_input.num_experts,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = policy_input.base_epoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .initial_owner_participants = initialOwners(policy_input),
                .minimum_window_activations =
                    policy_input.minimum_window_activations,
                .maximum_cycles_per_wave =
                    policy_input.maximum_cycles_per_wave,
                .dynamic_imbalance_threshold_per_mille =
                    policy_input.dynamic_imbalance_threshold_per_mille,
                .dynamic_minimum_improvement_per_mille =
                    policy_input.dynamic_minimum_improvement_per_mille,
                .dynamic_maximum_cycles_per_layer =
                    policy_input.dynamic_maximum_cycles_per_layer,
                .dynamic_maximum_commands_per_wave =
                    policy_input.dynamic_maximum_commands_per_wave,
            };
        };
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(0));
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(1));
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);

        ASSERT_NE(cuda_rank, nullptr);
        ASSERT_NE(rocm_rank, nullptr);
        publishSyntheticEconomy(
            *cuda_rank,
            *rocm_rank,
            *resolved_topology,
            policy_input.num_layers);
        const auto participant_bindings =
            allParticipantBindings(*cuda_rank, *rocm_rank);
        const auto transport_bindings =
            allTransportBindings(*cuda_rank, *rocm_rank);
        ASSERT_EQ(
            participant_bindings.size(),
            resolved_topology->participants.size());
        ASSERT_EQ(
            transport_bindings.size(), resolved_topology->groups.size());

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding_it = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &binding)
                {
                    return binding.participant_id ==
                           participant.participant_id;
                });
            ASSERT_NE(binding_it, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding_it->controller->admission_epoch,
                binding_it->lifetime));
        }

        RuntimeCandidateApplyEvidence observed;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &observed,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .exercise_reader_grace_period = true})) << error;

        EXPECT_EQ(
            observed.old_epoch_readers_acquired,
            resolved_topology->participants.size());
        EXPECT_EQ(
            observed.new_epoch_readers_completed,
            resolved_topology->participants.size());
        EXPECT_EQ(
            observed.retirement_readiness_deferrals_observed,
            resolved_topology->participants.size());
        EXPECT_EQ(
            observed.old_epoch_readers_released,
            resolved_topology->participants.size());
        EXPECT_EQ(
            observed.post_readiness_acquisition_guards,
            resolved_topology->participants.size());

        EXPECT_EQ(
            observed.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete));
        EXPECT_EQ(
            observed.controller.error_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None));
        EXPECT_EQ(
            observed.controller.current_durable_epoch,
            policy_input.base_epoch + 1u);
        EXPECT_EQ(
            observed.controller.admission_epoch,
            policy_input.base_epoch + 1u);
        EXPECT_EQ(
            observed.controller.candidate_epoch,
            policy_input.base_epoch + 1u);
        EXPECT_EQ(
            observed.controller.placement_layer_cursor,
            expected.evidence.layer_scan_next);
        EXPECT_EQ(observed.policy.command_count, expected.commands.size());
        EXPECT_EQ(
            observed.policy.command_digest,
            commandDigest(
                expected.commands.data(),
                static_cast<std::uint32_t>(expected.commands.size())));
        EXPECT_EQ(
            observed.policy.packed_weight_bytes,
            std::accumulate(
                expected.commands.begin(),
                expected.commands.end(),
                std::uint64_t{0u},
                [](std::uint64_t total, const auto &command)
                {
                    return total + command.payload_bytes;
                }));
        EXPECT_EQ(
            observed.policy.snapshot_observations,
            expected.evidence.snapshot_observations);
        EXPECT_EQ(
            observed.policy.priority_cost_before,
            expected.evidence.priority_cost_before);
        EXPECT_EQ(
            observed.policy.priority_cost_after,
            expected.evidence.priority_cost_after);
        EXPECT_EQ(
            observed.policy.same_priority_makespan_before,
            expected.evidence.same_priority_makespan_before);
        EXPECT_EQ(
            observed.policy.same_priority_makespan_after,
            expected.evidence.same_priority_makespan_after);
        EXPECT_EQ(
            observed.policy.accepted_cycles,
            expected.evidence.accepted_cycles);
        EXPECT_EQ(
            observed.policy.rejected_cycles,
            expected.evidence.rejected_cycles);
        EXPECT_EQ(observed.policy.promotions, expected.evidence.promotions);
        EXPECT_EQ(observed.policy.demotions, expected.evidence.demotions);
        EXPECT_EQ(
            observed.policy.same_priority_moves,
            expected.evidence.same_priority_moves);
        EXPECT_EQ(
            observed.policy.changed_layers,
            expected.evidence.changed_layers);
        EXPECT_EQ(
            observed.policy.layer_scan_start,
            expected.evidence.layer_scan_start);
        EXPECT_EQ(
            observed.policy.layer_scan_next,
            expected.evidence.layer_scan_next);
        EXPECT_EQ(
            observed.policy.projected_service_gain_ns,
            expected.evidence.projected_service_gain_ns);
        EXPECT_EQ(
            observed.policy.projected_transfer_and_repack_ns,
            expected.evidence.projected_transfer_and_repack_ns);
        EXPECT_EQ(
            observed.policy.projected_inference_interference_ns,
            expected.evidence.projected_inference_interference_ns);
        EXPECT_EQ(
            observed.policy.projected_net_benefit_ns,
            expected.evidence.projected_net_benefit_ns);
        EXPECT_EQ(
            observed.policy.payoff_rejected_cycles,
            expected.evidence.payoff_rejected_cycles);
        EXPECT_EQ(
            observed.policy.residency_rejected_cycles,
            expected.evidence.residency_rejected_cycles);

        ASSERT_EQ(observed.commands.size(), expected.commands.size());
        for (std::size_t index = 0u; index < expected.commands.size(); ++index)
        {
            EXPECT_EQ(
                std::memcmp(
                    &observed.commands[index],
                    &expected.commands[index],
                    sizeof(MoEOverlayDeviceMovementCommand)),
                0) << "device/CPU command ABI mismatch at ordinal " << index;
        }
        ASSERT_EQ(
            observed.economy_last_moved.size(),
            static_cast<std::size_t>(policy_input.num_layers) *
                policy_input.num_experts);
        std::vector<bool> moved(observed.economy_last_moved.size(), false);
        for (const auto &command : observed.commands)
        {
            moved[static_cast<std::size_t>(command.layer) *
                      policy_input.num_experts +
                  command.expert] = true;
        }
        for (std::size_t offset = 0u;
             offset < observed.economy_last_moved.size();
             ++offset)
        {
            EXPECT_EQ(
                observed.economy_last_moved[offset],
                moved[offset]
                    ? observed.controller.transaction_id
                    : kMoEOverlayDeviceControllerNeverMovedGeneration)
                << "committed hysteresis mismatch at layer/expert offset "
                << offset;
        }
        EXPECT_EQ(
            observed.transport_groups_prepared,
            transport_bindings.size());
        EXPECT_EQ(
            observed.transport_groups_published,
            transport_bindings.size());
        EXPECT_EQ(
            observed.transport_groups_retired,
            transport_bindings.size());

        ASSERT_EQ(observed.apply_statuses.size(), fixtures.size());
        ASSERT_EQ(observed.runtime_layers.size(), fixtures.size());
        ASSERT_EQ(observed.epoch_controls.size(), fixtures.size());
        ASSERT_EQ(observed.epoch_statuses.size(), fixtures.size());
        for (std::size_t participant = 0u;
             participant < fixtures.size();
             ++participant)
        {
            const auto &apply = observed.apply_statuses[participant];
            EXPECT_EQ(
                apply.code,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceRuntimeApplyCode::Success));
            EXPECT_EQ(apply.transaction_id, observed.controller.transaction_id);
            EXPECT_EQ(apply.base_epoch, policy_input.base_epoch);
            EXPECT_EQ(
                apply.candidate_epoch, policy_input.base_epoch + 1u);
            EXPECT_EQ(apply.candidate_bank, 1u);

            for (const auto &runtime : observed.runtime_layers[participant])
            {
                EXPECT_EQ(runtime.active_bank, 1u);
                EXPECT_EQ(
                    runtime.active_epoch, policy_input.base_epoch + 1u);
                EXPECT_EQ(
                    runtime.banks[1].epoch, policy_input.base_epoch + 1u);
            }

            for (const auto &entry : observed.commands)
            {
                const auto &runtime =
                    observed.runtime_layers[participant].at(entry.layer);
                const auto &descriptor =
                    runtime.banks[1].experts[entry.expert];
                const bool exact_destination =
                    entry.destination_participant == participant;
                if (exact_destination)
                {
                    EXPECT_EQ(
                        descriptor.logical_expert_id,
                        static_cast<std::int32_t>(entry.expert));
                    EXPECT_EQ(
                        descriptor.local_slot,
                        static_cast<std::int32_t>(entry.expert));
                    EXPECT_TRUE(descriptor.weightsReady());
                    EXPECT_TRUE(hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::Valid));
                    EXPECT_TRUE(hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::Resident));
                    EXPECT_TRUE(hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::LocalCompute));
                    EXPECT_FALSE(hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::TransferSlot))
                        << "Durable ExpertOverlay movement must not claim "
                           "request-scoped transfer-directory storage";
                }
                else if (entry.source_participant == participant)
                {
                    EXPECT_EQ(descriptor.local_slot, -1);
                    EXPECT_FALSE(hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::LocalCompute));
                }
            }

            const auto &control = observed.epoch_controls[participant];
            EXPECT_EQ(control.bank_epochs[0], 0u);
            EXPECT_EQ(control.bank_epochs[1], policy_input.base_epoch + 1u);
            EXPECT_EQ(control.bank_readers[0], 0u);
            EXPECT_EQ(control.bank_readers[1], 0u);
            EXPECT_EQ(control.acquisitions_in_flight, 0u);
            EXPECT_EQ(
                control.bank_states[0],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Empty));
            EXPECT_EQ(
                control.bank_states[1],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Published));
            EXPECT_EQ(
                deviceMoEOverlayEpochSelectorBank(
                    control.published_selector),
                1u);
            EXPECT_EQ(
                deviceMoEOverlayEpochSelectorGeneration(
                    control.published_selector),
                2u);

            const auto &status = observed.epoch_statuses[participant];
            EXPECT_EQ(
                status.operation,
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochOperation::Retire));
            EXPECT_EQ(
                status.code,
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochStatusCode::Success));
            EXPECT_EQ(status.epoch, policy_input.base_epoch);
            EXPECT_EQ(status.bank, 0u);
        }

        ASSERT_EQ(
            observed.participant_records.size(),
            resolved_topology->participants.size());
        for (const auto &record : observed.participant_records)
        {
            EXPECT_EQ(record.magic, kMoEOverlayDeviceControllerFabricMagic);
            EXPECT_EQ(record.version, kMoEOverlayDeviceControllerFabricVersion);
            EXPECT_EQ(
                record.topology_fingerprint,
                resolved_topology->topology_fingerprint);
            EXPECT_EQ(
                record.prepared_transaction,
                observed.controller.transaction_id);
            EXPECT_EQ(
                record.published_transaction,
                observed.controller.transaction_id);
            EXPECT_EQ(record.retired_epoch, policy_input.base_epoch);
            EXPECT_EQ(record.restored_transaction, 0u);
        }
    }

    /**
     * @brief Restore a moved CUDA/ROCm runtime to its immutable prepared owners.
     *
     * The test executes the exact retained durable transaction family three
     * times: one economical Dynamic movement, one inverse restoration wave,
     * and one zero-command restoration certification. The terminal proof
     * checks every active runtime descriptor against the loader-era owner
     * table and also proves restoration did not rewrite optimization
     * hysteresis or its placement scan cursor.
     */
    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceAuthoredPreparedContextRestoreReturnsEveryRuntimeOwnerExactly)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto policy_input = adversarialPolicyInput();
        const auto prepared_owners = initialOwners(policy_input);
        const auto expected_dynamic =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(expected_dynamic.hasMovement());

        auto fabrics = makeControllerFabricPair(
            resolved_topology, policy_input);
        publishSyntheticEconomy(
            *fabrics.cuda_rank,
            *fabrics.rocm_rank,
            *resolved_topology,
            policy_input.num_layers);
        const auto participant_bindings = allParticipantBindings(
            *fabrics.cuda_rank, *fabrics.rocm_rank);
        const auto transport_bindings = allTransportBindings(
            *fabrics.cuda_rank, *fabrics.rocm_rank);

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &candidate)
                {
                    return candidate.participant_id ==
                        participant.participant_id;
                });
            ASSERT_NE(binding, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding->controller->admission_epoch,
                binding->lifetime));
        }

        RuntimeCandidateApplyEvidence dynamic;
        RuntimeCandidateApplyEvidence restored;
        RuntimeCandidateApplyEvidence certified;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &dynamic,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
            })) << error;
        ASSERT_FALSE(dynamic.commands.empty());
        ASSERT_EQ(
            dynamic.command.kind,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::
                    DynamicPlacement));

        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &restored,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .objective = RuntimeCandidateApplyObjective::
                    PreparedContextRestore,
            })) << error;
        ASSERT_FALSE(restored.commands.empty());
        EXPECT_EQ(
            restored.command.kind,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::
                    PreparedContextRestore));
        EXPECT_EQ(
            restored.command.demand_phase,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceDemandPhase::Invalid));
        EXPECT_EQ(
            restored.controller.current_durable_epoch,
            policy_input.base_epoch + 2u);
        EXPECT_EQ(
            restored.controller.admission_epoch,
            policy_input.base_epoch + 2u);
        EXPECT_EQ(restored.command.projected_service_gain_ns, 0u);
        EXPECT_EQ(restored.command.projected_transfer_and_repack_ns, 0u);
        EXPECT_EQ(restored.command.projected_inference_interference_ns, 0u);
        EXPECT_EQ(restored.command.projected_net_benefit_ns, 0);
        EXPECT_EQ(
            restored.economy_last_moved,
            dynamic.economy_last_moved)
            << "Prepared-context repair cannot become Dynamic hysteresis";
        EXPECT_EQ(
            restored.controller.placement_layer_cursor,
            dynamic.controller.placement_layer_cursor)
            << "Prepared-context repair cannot advance the policy scan";

        ASSERT_EQ(restored.commands.size(), dynamic.commands.size());
        for (const auto &moved : dynamic.commands)
        {
            const auto inverse = std::find_if(
                restored.commands.begin(),
                restored.commands.end(),
                [&moved](const auto &candidate)
                {
                    return candidate.layer == moved.layer &&
                        candidate.expert == moved.expert &&
                        candidate.source_participant ==
                            moved.destination_participant &&
                        candidate.destination_participant ==
                            moved.source_participant;
                });
            EXPECT_NE(inverse, restored.commands.end())
                << "Missing inverse restoration for layer " << moved.layer
                << " expert " << moved.expert;
        }

        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &certified,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .expect_no_movement = true,
                .objective = RuntimeCandidateApplyObjective::
                    PreparedContextRestore,
            })) << error;
        EXPECT_TRUE(certified.commands.empty());
        EXPECT_EQ(
            certified.command.kind,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransactionKind::
                    PreparedContextRestore));
        EXPECT_EQ(
            certified.command.base_epoch,
            restored.controller.current_durable_epoch);
        EXPECT_EQ(
            certified.command.candidate_epoch,
            restored.controller.current_durable_epoch);
        EXPECT_EQ(
            certified.controller.current_durable_epoch,
            restored.controller.current_durable_epoch);
        EXPECT_EQ(
            certified.controller.completed_transaction,
            certified.controller.transaction_id);
        EXPECT_EQ(
            certified.economy_last_moved,
            dynamic.economy_last_moved);
        EXPECT_EQ(
            certified.controller.placement_layer_cursor,
            dynamic.controller.placement_layer_cursor);
        EXPECT_EQ(certified.transport_groups_prepared, 0u);
        EXPECT_EQ(certified.transport_groups_published, 0u);
        EXPECT_EQ(certified.transport_groups_retired, 0u);

        ASSERT_EQ(certified.runtime_layers.size(), fixtures.size());
        for (std::uint32_t layer = 0u;
             layer < policy_input.num_layers;
             ++layer)
        {
            for (std::uint32_t expert = 0u;
                 expert < policy_input.num_experts;
                 ++expert)
            {
                const std::uint32_t expected_owner = prepared_owners[
                    static_cast<std::size_t>(layer) *
                        policy_input.num_experts +
                    expert];
                std::uint32_t active_owner_count = 0u;
                for (std::size_t fixture_index = 0u;
                     fixture_index < fixtures.size();
                     ++fixture_index)
                {
                    const auto &runtime =
                        certified.runtime_layers[fixture_index].at(layer);
                    ASSERT_LT(
                        runtime.active_bank,
                        kDeviceMoEOverlayEpochBankCount);
                    EXPECT_EQ(
                        runtime.active_epoch,
                        restored.controller.current_durable_epoch);
                    const auto &descriptor =
                        runtime.banks[runtime.active_bank].experts[expert];
                    const bool owns = hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::LocalCompute);
                    if (!owns)
                        continue;
                    ++active_owner_count;
                    EXPECT_EQ(
                        fixtures[fixture_index]->participantId(),
                        static_cast<int>(expected_owner));
                    EXPECT_EQ(
                        descriptor.logical_expert_id,
                        static_cast<std::int32_t>(expert));
                    EXPECT_TRUE(descriptor.weightsReady());
                }
                EXPECT_EQ(active_owner_count, 1u)
                    << "Prepared owner multiplicity mismatch at layer "
                    << layer << " expert " << expert;
            }
        }
    }

    /**
     * @brief Restore accumulated drift without borrowing another layer's slots.
     * @param authority_backend Backend owning the captured placement author.
     *
     * Seven disjoint six-participant cycles in each of two layers model the
     * result of many earlier bounded Dynamic waves. The prepared owner table
     * and live runtime are deliberately different at construction. Repair must
     * use multiple epochs, while independent layers still share each wave.
     * Synthetic weight descriptors isolate publication from payload kernels;
     * the real-model campaign separately verifies the physical byte transfers.
     */
    void proveBoundedPreparedContextRestore(DeviceType authority_backend)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 || rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }
        const auto resolved = topology(authority_backend);
        constexpr std::uint32_t cycles_per_layer = 7u;
        const auto participants = static_cast<std::uint32_t>(resolved->participants.size());
        for (const std::uint32_t policy_layer_limit : {0u, 1u, 2u, 4u})
        {
            SCOPED_TRACE(policy_layer_limit);
            // A zero optimization cap cannot disable terminal restoration;
            // admission still owns one cycle's physical capacity in this mode.
            const auto layer_limit = std::max(1u, policy_layer_limit);
            auto prepared = adversarialPolicyInput();
            prepared.num_experts = participants * cycles_per_layer;
            prepared.dynamic_maximum_cycles_per_layer = policy_layer_limit;
            prepared.economy.reset(); // Lifecycle repair has no economy objective.
            const auto plane = prepared.num_layers * prepared.num_experts;
            prepared.collected_state.assign(participants * plane, 0u);
            auto live = prepared;
            for (const auto &participant : resolved->participants)
            {
                auto &metadata = prepared.participants.at(participant.participant_id);
                metadata.tier_index = participant.tier_idx;
                metadata.tier_priority = resolved->groupForParticipant(
                    participant.participant_id)->tier_priority;
            }
            live.participants = prepared.participants;
            for (std::uint32_t layer = 0u; layer < prepared.num_layers; ++layer)
            {
                for (std::uint32_t expert = 0u; expert < prepared.num_experts; ++expert)
                {
                    const auto initial = expert % participants;
                    const auto moved = (initial + (layer == 0u ? 1u : participants - 1u)) % participants;
                    const auto word = moe_rebalance_policy::packCollectedState(
                        1u, 0u, true, false, true);
                    prepared.collected_state[initial * plane + layer * prepared.num_experts + expert] = word;
                    live.collected_state[moved * plane + layer * prepared.num_experts + expert] = word;
                }
            }
            auto fabrics = makeControllerFabricPair(resolved, prepared);
            const auto bindings = allParticipantBindings(*fabrics.cuda_rank, *fabrics.rocm_rank);
            const auto transports = allTransportBindings(*fabrics.cuda_rank, *fabrics.rocm_rank);
            std::vector<std::unique_ptr<MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
            for (const auto &participant : resolved->participants)
            {
                const auto binding = std::find_if(bindings.begin(), bindings.end(),
                    [&participant](const auto &entry)
                    { return entry.participant_id == participant.participant_id; });
                ASSERT_NE(binding, bindings.end());
                fixtures.push_back(std::make_unique<MoEOverlayDeviceRuntimePublicationFixture>(
                    participant.device.type == DeviceType::CUDA ? cuda : rocm,
                    participant, *resolved, live, &binding->controller->admission_epoch,
                    binding->lifetime));
            }

            auto owners = initialOwners(live);
            const auto targets = initialOwners(prepared);
            const auto waves = (cycles_per_layer + layer_limit - 1u) / layer_limit;
            RuntimeCandidateApplyEvidence observed;
            std::vector<std::uint64_t> movement_history;
            std::string error;
            for (std::uint32_t wave = 0u; wave <= waves; ++wave)
            {
                SCOPED_TRACE(wave);
                const bool terminal = wave == waves;
                ASSERT_TRUE(runRuntimeCandidateApply(cuda, rocm, *resolved, bindings,
                    transports, fixtures, &observed, &error,
                    {.complete_epoch = true, .expect_no_movement = terminal,
                     .objective = RuntimeCandidateApplyObjective::PreparedContextRestore})) << error;
                EXPECT_EQ(observed.command.demand_phase,
                    static_cast<std::uint32_t>(MoEOverlayDeviceDemandPhase::Invalid));
                EXPECT_EQ(observed.command.projected_net_benefit_ns, 0);
                if (wave == 0u) movement_history = observed.economy_last_moved;
                EXPECT_EQ(observed.economy_last_moved, movement_history);
                if (terminal)
                {
                    EXPECT_TRUE(observed.commands.empty());
                    EXPECT_EQ(owners, targets);
                    EXPECT_EQ(observed.controller.current_durable_epoch, prepared.base_epoch + waves);
                    break;
                }
                const auto cycles = std::min(layer_limit, cycles_per_layer - wave * layer_limit);
                ASSERT_EQ(observed.commands.size(), prepared.num_layers * participants * cycles)
                    << "One layer must not consume the global wave's shadow capacity";
                EXPECT_EQ(observed.policy.changed_layers, prepared.num_layers)
                    << "Independent layers must retain parallel wave admission";
                std::vector<std::uint32_t> arrivals(participants * prepared.num_layers, 0u);
                for (const auto &command : observed.commands)
                {
                    ASSERT_LT(command.layer, prepared.num_layers);
                    ASSERT_LT(command.expert, prepared.num_experts);
                    ASSERT_LT(command.destination_participant, participants);
                    const auto index = command.layer * prepared.num_experts + command.expert;
                    EXPECT_EQ(command.source_participant, owners[index]);
                    EXPECT_EQ(command.destination_participant, targets[index]);
                    EXPECT_NE(owners[index], targets[index]) << "Repair must make strict progress";
                    owners[index] = command.destination_participant;
                    EXPECT_LE(++arrivals[command.destination_participant * prepared.num_layers + command.layer], layer_limit);
                }
            }
            // The terminal zero-command receipt certifies live device tables,
            // not just the host's independent expected-command reconstruction.
            for (std::uint32_t layer = 0u; layer < prepared.num_layers; ++layer)
                for (std::uint32_t expert = 0u; expert < prepared.num_experts; ++expert)
                {
                    std::uint32_t owner_count = 0u;
                    for (std::size_t p = 0u; p < fixtures.size(); ++p)
                    {
                        const auto &runtime = observed.runtime_layers.at(p).at(layer);
                        ASSERT_LT(runtime.active_bank, kDeviceMoEOverlayEpochBankCount);
                        const auto &descriptor = runtime.banks[runtime.active_bank].experts[expert];
                        if (!hasMoEExpertFlag(descriptor.flags, DeviceMoEExpertFlags::LocalCompute)) continue;
                        ++owner_count;
                        EXPECT_EQ(fixtures[p]->participantId(), targets[layer * prepared.num_experts + expert]);
                        EXPECT_TRUE(descriptor.weightsReady());
                    }
                    EXPECT_EQ(owner_count, 1u);
                }
        }
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CUDAAuthoredPreparedRestoreHonorsLayerCapacityAcrossEpochs)
    {
        proveBoundedPreparedContextRestore(DeviceType::CUDA);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ROCmAuthoredPreparedRestoreHonorsLayerCapacityAcrossEpochs)
    {
        proveBoundedPreparedContextRestore(DeviceType::ROCm);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceAuthoredDynamicNoMovementCompletesWithoutDescriptorOrEpochMutation)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        auto policy_input = adversarialPolicyInput();
        policy_input.minimum_window_activations =
            std::numeric_limits<std::uint64_t>::max();
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_FALSE(expected.hasMovement());

        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        const auto fabric_config = [&](int rank)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = context(rank),
                .topology = resolved_topology,
                .num_layers = policy_input.num_layers,
                .num_experts = policy_input.num_experts,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = policy_input.base_epoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .initial_owner_participants = initialOwners(policy_input),
                .minimum_window_activations =
                    policy_input.minimum_window_activations,
                .maximum_cycles_per_wave =
                    policy_input.maximum_cycles_per_wave,
                .dynamic_imbalance_threshold_per_mille =
                    policy_input.dynamic_imbalance_threshold_per_mille,
                .dynamic_minimum_improvement_per_mille =
                    policy_input.dynamic_minimum_improvement_per_mille,
                .dynamic_maximum_cycles_per_layer =
                    policy_input.dynamic_maximum_cycles_per_layer,
                .dynamic_maximum_commands_per_wave =
                    policy_input.dynamic_maximum_commands_per_wave,
            };
        };
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(0));
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(1));
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);

        ASSERT_NE(cuda_rank, nullptr);
        ASSERT_NE(rocm_rank, nullptr);
        publishSyntheticEconomy(
            *cuda_rank,
            *rocm_rank,
            *resolved_topology,
            policy_input.num_layers);
        const auto participant_bindings =
            allParticipantBindings(*cuda_rank, *rocm_rank);
        const auto transport_bindings =
            allTransportBindings(*cuda_rank, *rocm_rank);
        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &candidate)
                {
                    return candidate.participant_id ==
                        participant.participant_id;
                });
            ASSERT_NE(binding, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding->controller->admission_epoch,
                binding->lifetime));
        }

        RuntimeCandidateApplyEvidence observed;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &observed,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .expect_no_movement = true})) << error;

        EXPECT_EQ(
            observed.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete));
        EXPECT_EQ(
            observed.controller.error_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None));
        EXPECT_EQ(
            observed.controller.current_durable_epoch,
            policy_input.base_epoch);
        EXPECT_EQ(
            observed.controller.admission_epoch,
            policy_input.base_epoch);
        EXPECT_EQ(observed.command.command_count, 0u);
        EXPECT_EQ(observed.command.packed_weight_bytes, 0u);
        EXPECT_EQ(observed.policy.command_count, 0u);
        EXPECT_EQ(observed.transport_groups_prepared, 0u);
        EXPECT_EQ(observed.transport_groups_published, 0u);
        EXPECT_EQ(observed.transport_groups_retired, 0u);

        for (std::size_t participant = 0u;
             participant < fixtures.size();
             ++participant)
        {
            EXPECT_EQ(
                observed.apply_statuses[participant].code,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceRuntimeApplyCode::Idle));
            for (const auto &runtime : observed.runtime_layers[participant])
            {
                EXPECT_EQ(runtime.active_bank, 0u);
                EXPECT_EQ(runtime.active_epoch, policy_input.base_epoch);
            }
            const auto &control = observed.epoch_controls[participant];
            EXPECT_EQ(control.bank_epochs[0], policy_input.base_epoch);
            EXPECT_EQ(control.bank_epochs[1], 0u);
            EXPECT_EQ(
                control.bank_states[0],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Published));
            EXPECT_EQ(
                control.bank_states[1],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Empty));
        }
        for (const auto &record : observed.participant_records)
        {
            EXPECT_EQ(record.prepared_transaction, 0u);
            EXPECT_EQ(record.published_transaction, 0u);
            EXPECT_EQ(record.retired_epoch, 0u);
        }
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         FailedArrivalCannotPublishOrRetireAnyCudaOrRocmRuntimeEpoch)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto policy_input = adversarialPolicyInput();
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(expected.hasMovement());

        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        const auto fabric_config = [&](int rank)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = context(rank),
                .topology = resolved_topology,
                .num_layers = policy_input.num_layers,
                .num_experts = policy_input.num_experts,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = policy_input.base_epoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .initial_owner_participants = initialOwners(policy_input),
                .minimum_window_activations =
                    policy_input.minimum_window_activations,
                .maximum_cycles_per_wave =
                    policy_input.maximum_cycles_per_wave,
                .dynamic_imbalance_threshold_per_mille =
                    policy_input.dynamic_imbalance_threshold_per_mille,
                .dynamic_minimum_improvement_per_mille =
                    policy_input.dynamic_minimum_improvement_per_mille,
                .dynamic_maximum_cycles_per_layer =
                    policy_input.dynamic_maximum_cycles_per_layer,
                .dynamic_maximum_commands_per_wave =
                    policy_input.dynamic_maximum_commands_per_wave,
            };
        };
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(0));
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(1));
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);
        ASSERT_NE(cuda_rank, nullptr);
        ASSERT_NE(rocm_rank, nullptr);

        publishSyntheticEconomy(
            *cuda_rank,
            *rocm_rank,
            *resolved_topology,
            policy_input.num_layers);

        const auto participant_bindings =
            allParticipantBindings(*cuda_rank, *rocm_rank);
        const auto transport_bindings =
            allTransportBindings(*cuda_rank, *rocm_rank);
        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding_it = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &binding)
                {
                    return binding.participant_id ==
                           participant.participant_id;
                });
            ASSERT_NE(binding_it, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding_it->controller->admission_epoch,
                binding_it->lifetime));
        }

        RuntimeCandidateApplyEvidence observed;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &observed,
            &error,
            RuntimeCandidateApplyOptions{
                .complete_epoch = true,
                .fault = RuntimeCandidateApplyFault::
                    OmitFirstDestinationArrival})) << error;

        EXPECT_EQ(
            observed.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Error));
        EXPECT_NE(
            observed.controller.error_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None));
        EXPECT_EQ(
            observed.controller.current_durable_epoch,
            policy_input.base_epoch);
        EXPECT_EQ(
            observed.controller.admission_epoch,
            policy_input.base_epoch);
        EXPECT_EQ(observed.controller.commit_transaction, 0u);
        EXPECT_EQ(
            observed.transport_groups_prepared,
            transport_bindings.size());
        EXPECT_EQ(observed.transport_groups_published, 0u);
        EXPECT_EQ(observed.transport_groups_retired, 0u);

        ASSERT_EQ(observed.apply_statuses.size(), fixtures.size());
        ASSERT_EQ(observed.runtime_layers.size(), fixtures.size());
        ASSERT_EQ(observed.epoch_controls.size(), fixtures.size());
        ASSERT_EQ(observed.epoch_statuses.size(), fixtures.size());
        EXPECT_TRUE(std::any_of(
            observed.apply_statuses.begin(),
            observed.apply_statuses.end(),
            [](const auto &status)
            {
                return status.code == static_cast<std::uint32_t>(
                                          MoEOverlayDeviceRuntimeApplyCode::
                                              MissingArrival);
            }));
        for (std::size_t participant = 0u;
             participant < fixtures.size();
             ++participant)
        {
            // Other participants may have completed their inactive-bank copy
            // before the missing destination poisons the topology. None may
            // flip its selector or canonical runtime bank afterward.
            for (const auto &runtime : observed.runtime_layers[participant])
            {
                EXPECT_EQ(runtime.active_bank, 0u);
                EXPECT_EQ(runtime.active_epoch, policy_input.base_epoch);
                EXPECT_EQ(runtime.banks[0].epoch, policy_input.base_epoch);
            }
            const auto &control = observed.epoch_controls[participant];
            EXPECT_EQ(
                deviceMoEOverlayEpochSelectorBank(
                    control.published_selector),
                0u);
            EXPECT_EQ(
                deviceMoEOverlayEpochSelectorGeneration(
                    control.published_selector),
                1u);
            EXPECT_EQ(control.bank_epochs[0], policy_input.base_epoch);
            EXPECT_EQ(
                control.bank_states[0],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Published));
            EXPECT_NE(
                control.bank_states[1],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Published));
            EXPECT_NE(
                control.bank_states[1],
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochBankState::Retiring));
            EXPECT_NE(
                observed.epoch_statuses[participant].operation,
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochOperation::PublishCandidate));
            EXPECT_NE(
                observed.epoch_statuses[participant].operation,
                static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochOperation::Retire));
        }

        ASSERT_EQ(
            observed.participant_records.size(),
            resolved_topology->participants.size());
        for (const auto &record : observed.participant_records)
        {
            EXPECT_EQ(record.published_transaction, 0u);
            EXPECT_EQ(record.retired_epoch, 0u);
        }
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceAuthoredDynamicBuildsEveryCudaAndRocmInactiveRuntimeBank)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto policy_input = adversarialPolicyInput();
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(expected.hasMovement());
        ASSERT_GT(expected.evidence.promotions, 0u);
        ASSERT_GT(expected.evidence.demotions, 0u);
        ASSERT_GT(expected.evidence.same_priority_moves, 0u);

        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        const auto fabric_config = [&](int rank)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = context(rank),
                .topology = resolved_topology,
                .num_layers = policy_input.num_layers,
                .num_experts = policy_input.num_experts,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = policy_input.base_epoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .initial_owner_participants = initialOwners(policy_input),
                .minimum_window_activations =
                    policy_input.minimum_window_activations,
                .maximum_cycles_per_wave =
                    policy_input.maximum_cycles_per_wave,
                .dynamic_imbalance_threshold_per_mille =
                    policy_input.dynamic_imbalance_threshold_per_mille,
                .dynamic_minimum_improvement_per_mille =
                    policy_input.dynamic_minimum_improvement_per_mille,
                .dynamic_maximum_cycles_per_layer =
                    policy_input.dynamic_maximum_cycles_per_layer,
                .dynamic_maximum_commands_per_wave =
                    policy_input.dynamic_maximum_commands_per_wave,
            };
        };
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(0));
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             fabric_config(1));
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);
        ASSERT_NE(cuda_rank, nullptr);
        ASSERT_NE(rocm_rank, nullptr);

        publishSyntheticEconomy(
            *cuda_rank,
            *rocm_rank,
            *resolved_topology,
            policy_input.num_layers);

        const auto participant_bindings =
            allParticipantBindings(*cuda_rank, *rocm_rank);
        const auto transport_bindings =
            allTransportBindings(*cuda_rank, *rocm_rank);
        ASSERT_EQ(
            participant_bindings.size(),
            resolved_topology->participants.size());
        ASSERT_EQ(
            transport_bindings.size(), resolved_topology->groups.size());

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            const auto binding_it = std::find_if(
                participant_bindings.begin(),
                participant_bindings.end(),
                [&participant](const auto &binding)
                {
                    return binding.participant_id ==
                           participant.participant_id;
                });
            ASSERT_NE(binding_it, participant_bindings.end());
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                backend,
                participant,
                *resolved_topology,
                policy_input,
                &binding_it->controller->admission_epoch,
                binding_it->lifetime));
        }

        RuntimeCandidateApplyEvidence observed;
        std::string error;
        ASSERT_TRUE(runRuntimeCandidateApply(
            cuda,
            rocm,
            *resolved_topology,
            participant_bindings,
            transport_bindings,
            fixtures,
            &observed,
            &error)) << error;

        EXPECT_EQ(
            observed.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::PreparingFollowers));
        EXPECT_EQ(
            observed.controller.error_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None));
        EXPECT_EQ(
            observed.controller.current_durable_epoch,
            policy_input.base_epoch);
        EXPECT_EQ(observed.controller.admission_epoch, policy_input.base_epoch);
        EXPECT_EQ(
            observed.controller.candidate_epoch,
            policy_input.base_epoch + 1u);
        EXPECT_EQ(
            observed.controller.placement_layer_cursor,
            expected.evidence.layer_scan_next);
        EXPECT_EQ(observed.transport_groups_prepared, transport_bindings.size());

        EXPECT_EQ(observed.policy.command_count, expected.commands.size());
        EXPECT_EQ(
            observed.policy.command_digest,
            commandDigest(
                expected.commands.data(),
                static_cast<std::uint32_t>(expected.commands.size())));
        EXPECT_EQ(
            observed.policy.priority_cost_before,
            expected.evidence.priority_cost_before);
        EXPECT_EQ(
            observed.policy.priority_cost_after,
            expected.evidence.priority_cost_after);
        EXPECT_EQ(
            observed.policy.same_priority_makespan_before,
            expected.evidence.same_priority_makespan_before);
        EXPECT_EQ(
            observed.policy.same_priority_makespan_after,
            expected.evidence.same_priority_makespan_after);
        EXPECT_EQ(observed.policy.promotions, expected.evidence.promotions);
        EXPECT_EQ(observed.policy.demotions, expected.evidence.demotions);
        EXPECT_EQ(
            observed.policy.same_priority_moves,
            expected.evidence.same_priority_moves);
        EXPECT_EQ(
            observed.policy.layer_scan_start,
            expected.evidence.layer_scan_start);
        EXPECT_EQ(
            observed.policy.layer_scan_next,
            expected.evidence.layer_scan_next);
        EXPECT_EQ(
            observed.policy.projected_service_gain_ns,
            expected.evidence.projected_service_gain_ns);
        EXPECT_EQ(
            observed.policy.projected_transfer_and_repack_ns,
            expected.evidence.projected_transfer_and_repack_ns);
        EXPECT_EQ(
            observed.policy.projected_inference_interference_ns,
            expected.evidence.projected_inference_interference_ns);
        EXPECT_EQ(
            observed.policy.projected_net_benefit_ns,
            expected.evidence.projected_net_benefit_ns);
        ASSERT_EQ(observed.commands.size(), expected.commands.size());
        for (std::size_t index = 0u; index < expected.commands.size(); ++index)
        {
            EXPECT_EQ(
                std::memcmp(
                    &observed.commands[index],
                    &expected.commands[index],
                    sizeof(MoEOverlayDeviceMovementCommand)),
                0) << "device/CPU command ABI mismatch at ordinal " << index;
        }
        EXPECT_TRUE(std::all_of(
            observed.economy_last_moved.begin(),
            observed.economy_last_moved.end(),
            [](std::uint64_t generation)
            {
                return generation ==
                    kMoEOverlayDeviceControllerNeverMovedGeneration;
            })) << "pre-admission preparation mutated committed hysteresis";

        ASSERT_EQ(observed.apply_statuses.size(), fixtures.size());
        ASSERT_EQ(observed.runtime_layers.size(), fixtures.size());
        const std::uint32_t expected_flags = toMoEExpertFlags(
            DeviceMoEExpertFlags::Valid |
            DeviceMoEExpertFlags::Resident |
            DeviceMoEExpertFlags::PreferredOwner |
            DeviceMoEExpertFlags::LocalCompute);
        for (std::size_t fixture_index = 0u;
             fixture_index < fixtures.size();
             ++fixture_index)
        {
            const auto participant_id = fixtures[fixture_index]->participantId();
            const auto &participant = resolved_topology->participants.at(
                static_cast<std::size_t>(participant_id));
            const auto *const local_group =
                resolved_topology->groupForParticipant(participant_id);
            ASSERT_NE(local_group, nullptr);

            const auto &status = observed.apply_statuses[fixture_index];
            EXPECT_TRUE(status.succeeded())
                << "participant " << participant_id;
            EXPECT_EQ(status.transaction_id, 1u);
            EXPECT_EQ(status.base_epoch, policy_input.base_epoch);
            EXPECT_EQ(
                status.candidate_epoch, policy_input.base_epoch + 1u);
            EXPECT_EQ(status.candidate_bank, 1u);
            EXPECT_EQ(status.commands_observed, expected.commands.size());
            EXPECT_EQ(status.commands_applied, expected.commands.size());
            EXPECT_EQ(
                status.changed_layers, expected.evidence.changed_layers);
            EXPECT_EQ(status.missing_arrivals, 0u);
            EXPECT_EQ(status.invalid_runtime_layers, 0u);

            ASSERT_EQ(
                observed.runtime_layers[fixture_index].size(),
                policy_input.num_layers);
            for (std::uint32_t layer = 0u;
                 layer < policy_input.num_layers;
                 ++layer)
            {
                const auto &runtime =
                    observed.runtime_layers[fixture_index][layer];
                const auto &published = runtime.banks[0];
                const auto &candidate = runtime.banks[1];

                // Preparation is background work: the inference-visible bank
                // and epoch remain exactly the admitted base generation.
                EXPECT_EQ(runtime.active_bank, 0u);
                EXPECT_EQ(runtime.active_epoch, policy_input.base_epoch);
                EXPECT_EQ(published.epoch, policy_input.base_epoch);
                EXPECT_EQ(published.expert_count, policy_input.num_experts);
                EXPECT_EQ(
                    candidate.epoch, policy_input.base_epoch + 1u);
                EXPECT_EQ(candidate.expert_count, policy_input.num_experts);

                for (std::uint32_t expert = 0u;
                     expert < policy_input.num_experts;
                     ++expert)
                {
                    const auto command_it = std::find_if(
                        observed.commands.begin(),
                        observed.commands.end(),
                        [layer, expert](const auto &command)
                        {
                            return command.layer == layer &&
                                   command.expert == expert;
                        });
                    if (command_it == observed.commands.end())
                    {
                        EXPECT_EQ(
                            std::memcmp(
                                &candidate.experts[expert],
                                &published.experts[expert],
                                sizeof(DeviceMoEExpertDescriptor)),
                            0) << "untouched descriptor changed on participant "
                               << participant_id << ", layer " << layer
                               << ", expert " << expert;
                        EXPECT_EQ(
                            candidate.local_compute_mask[expert],
                            published.local_compute_mask[expert]);
                        EXPECT_EQ(
                            candidate.replica_role[expert],
                            published.replica_role[expert]);
                        EXPECT_EQ(
                            candidate.resident_participant_mask[expert],
                            published.resident_participant_mask[expert]);
                        EXPECT_EQ(
                            candidate.overlay_route_participant[expert],
                            published.overlay_route_participant[expert]);
                        continue;
                    }

                    const auto &command = *command_it;
                    const auto &destination =
                        resolved_topology->participants.at(
                            command.destination_participant);
                    const auto *const destination_group =
                        resolved_topology->groupForParticipant(
                            static_cast<int>(command.destination_participant));
                    ASSERT_NE(destination_group, nullptr);
                    const auto &descriptor = candidate.experts[expert];
                    EXPECT_EQ(
                        candidate.overlay_route_participant[expert],
                        static_cast<std::int32_t>(
                            command.destination_participant));

                    if (local_group->group_id == destination_group->group_id)
                    {
                        const auto local_destination =
                            static_cast<std::uint32_t>(
                                destination.domain_participant_index);
                        EXPECT_EQ(
                            descriptor.owner_participant,
                            static_cast<std::int32_t>(local_destination));
                        EXPECT_EQ(
                            candidate.resident_participant_mask[expert],
                            1u << local_destination);
                        if (participant_id == static_cast<int>(
                                                  command.destination_participant))
                        {
                            EXPECT_TRUE(descriptor.weightsReady());
                            // ExpertOverlay arrivals live in model-lifetime
                            // durable slots. Command ordinals and TransferSlot
                            // identify request-scoped LLEP/prefix payloads and
                            // must never leak into this runtime bank.
                            EXPECT_EQ(
                                descriptor.local_slot,
                                static_cast<std::int32_t>(expert));
                            EXPECT_EQ(descriptor.flags, expected_flags);
                            EXPECT_EQ(
                                candidate.local_compute_mask[expert], 1u);
                            EXPECT_EQ(
                                candidate.replica_role[expert],
                                static_cast<std::uint8_t>(
                                    DeviceMoEReplicaRole::Primary));
                        }
                        else
                        {
                            EXPECT_FALSE(descriptor.weightsReady());
                            EXPECT_EQ(descriptor.local_slot, -1);
                            EXPECT_EQ(descriptor.flags, 0u);
                            EXPECT_EQ(
                                candidate.local_compute_mask[expert], 0u);
                            EXPECT_EQ(
                                candidate.replica_role[expert],
                                static_cast<std::uint8_t>(
                                    DeviceMoEReplicaRole::None));
                        }
                    }
                    else
                    {
                        EXPECT_FALSE(descriptor.weightsReady());
                        EXPECT_EQ(descriptor.owner_participant, -1);
                        EXPECT_EQ(descriptor.local_slot, -1);
                        EXPECT_EQ(descriptor.flags, 0u);
                        EXPECT_EQ(
                            candidate.resident_participant_mask[expert], 0u);
                        EXPECT_EQ(candidate.local_compute_mask[expert], 0u);
                        EXPECT_EQ(
                            candidate.replica_role[expert],
                            static_cast<std::uint8_t>(
                                DeviceMoEReplicaRole::None));
                    }
                }
            }
        }
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         TwoCudaFourRocmRanksShareBothOwnerDirections)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_rank;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_rank;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             MoEOverlayNodeLocalDeviceControllerFabric::Config{
                                                 .mpi_ctx = context(0),
                                                 .topology = resolved_topology,
                                                 .num_layers = 2u,
                                                 .num_experts = 16u,
                                                 .command_capacity = 8u,
                                                 .initial_durable_epoch = 3u,
                                             });
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_rank = std::make_unique<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             MoEOverlayNodeLocalDeviceControllerFabric::Config{
                                                 .mpi_ctx = context(1),
                                                 .topology = resolved_topology,
                                                 .num_layers = 2u,
                                                 .num_experts = 16u,
                                                 .command_capacity = 8u,
                                                 .initial_durable_epoch = 3u,
                                             });
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);
        ASSERT_NE(cuda_rank, nullptr);
        ASSERT_NE(rocm_rank, nullptr);
        EXPECT_EQ(cuda_rank->localParticipantIds(),
                  (std::vector<int>{0, 1}));
        EXPECT_EQ(rocm_rank->localParticipantIds(),
                  (std::vector<int>{2, 3, 4, 5}));

        const auto cuda_leader = cuda_rank->participantBinding(0);
        const auto rocm_root = rocm_rank->participantBinding(2);
        const auto participant_bindings =
            allParticipantBindings(*cuda_rank, *rocm_rank);
        const auto transport_bindings =
            allTransportBindings(*cuda_rank, *rocm_rank);
        ASSERT_TRUE(cuda_leader.valid());
        ASSERT_TRUE(rocm_root.valid());
        EXPECT_TRUE(cuda_leader.authority_leader);
        EXPECT_TRUE(cuda_leader.group_root);
        EXPECT_FALSE(rocm_root.authority_leader);
        EXPECT_TRUE(rocm_root.group_root);

        const auto &rocm_group = rocm_rank->layout().groups.at(1u);
        void *const cuda_view_of_rocm_snapshot =
            cuda_leader.mapped_base_device +
            static_cast<std::size_t>(rocm_group.collected_state_offset);
        std::string error;
        ASSERT_TRUE(proveVisibility(
            rocm,
            /*producer_ordinal=*/0,
            rocm_root.group_collected_state,
            cuda,
            /*consumer_ordinal=*/0,
            cuda_view_of_rocm_snapshot,
            0x5a,
            &error)) << error;

        const auto command_offset = static_cast<std::size_t>(
            rocm_rank->layout().header.command_entries_offset);
        const void *const rocm_view_of_cuda_commands =
            rocm_root.mapped_base_device + command_offset;
        ASSERT_TRUE(proveVisibility(
            cuda,
            /*producer_ordinal=*/0,
            cuda_leader.command_entries,
            rocm,
            /*consumer_ordinal=*/0,
            rocm_view_of_cuda_commands,
            0xa6,
            &error)) << error;

        std::uint64_t expected_epoch = 3u;
        std::uint64_t expected_transaction = 0u;
        for (const auto kind : {
                 MoEOverlayDeviceControllerTransactionKind::StaticCheck,
                 MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP})
        {
            ControllerTransactionEvidence evidence;
            ASSERT_TRUE(runControllerTransaction(
                cuda,
                rocm,
                participant_bindings,
                transport_bindings,
                cuda_rank->layout(),
                kind,
                expected_epoch,
                &evidence,
                &error)) << error;

            ++expected_transaction;
            EXPECT_EQ(
                evidence.controller.state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::Complete));
            EXPECT_EQ(
                evidence.controller.error_code,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerError::None));
            EXPECT_EQ(evidence.controller.transaction_id, expected_transaction);
            EXPECT_EQ(evidence.controller.current_durable_epoch, expected_epoch);
            EXPECT_EQ(evidence.controller.admission_epoch, expected_epoch);
            EXPECT_EQ(evidence.controller.active_llep_transaction, 0u);
            EXPECT_EQ(
                evidence.command.kind,
                static_cast<std::uint32_t>(kind));
            EXPECT_EQ(
                evidence.command.command_count,
                kind ==
                        MoEOverlayDeviceControllerTransactionKind::StaticCheck
                    ? 0u
                    : 1u);
            EXPECT_EQ(
                evidence.command.packed_weight_bytes,
                kind == MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement
                    ? 4096u
                    : 0u);
            EXPECT_EQ(
                evidence.command.parallel_command_count,
                evidence.command.command_count);
            EXPECT_EQ(
                evidence.command.movement_round_count,
                evidence.command.command_count == 0u ? 0u : 1u);
            EXPECT_EQ(evidence.command.hazard_count, 0u);
            EXPECT_GT(evidence.leader_elapsed_ms, 0.0F);
            EXPECT_EQ(
                evidence.transport_groups_acquired,
                transport_bindings.size());
            const bool moves_weights =
                kind == MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement;
            EXPECT_EQ(
                evidence.transport_groups_prepared,
                moves_weights ? transport_bindings.size() : 0u);
            EXPECT_EQ(
                evidence.transport_groups_published,
                moves_weights ? transport_bindings.size() : 0u);
            EXPECT_EQ(
                evidence.transport_groups_retired,
                moves_weights ? transport_bindings.size() : 0u);
            EXPECT_EQ(evidence.transport_groups_restored, 0u);
            if (moves_weights)
                EXPECT_GE(evidence.leader_elapsed_ms, 1.0F);
            for (const auto &group : evidence.groups)
            {
                EXPECT_EQ(group.snapshot_transaction, expected_transaction);
                EXPECT_EQ(group.prepared_transaction, expected_transaction);
                EXPECT_EQ(group.published_transaction, expected_transaction);
                EXPECT_NE(group.snapshot_digest, 0u);
                EXPECT_EQ(
                    group.status_code,
                    static_cast<std::uint32_t>(
                        MoEOverlayDeviceControllerError::None));
                if (kind == MoEOverlayDeviceControllerTransactionKind::
                                DynamicPlacement)
                {
                    EXPECT_EQ(group.retired_epoch, expected_epoch - 1u);
                }
                if (kind == MoEOverlayDeviceControllerTransactionKind::
                                CurrentBatchLLEP)
                {
                    EXPECT_EQ(
                        group.restored_transaction,
                        expected_transaction);
                }
            }
        }

        // A third device-authored transaction deliberately reuses one
        // destination slot. The leader must poison the epoch before prepare,
        // even though the policy digest and total byte count are internally
        // consistent and both GPU families execute the complete chain.
        ControllerTransactionEvidence collision;
        ASSERT_TRUE(runControllerTransaction(
            cuda,
            rocm,
            participant_bindings,
            transport_bindings,
            cuda_rank->layout(),
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            expected_epoch,
            &collision,
            &error,
            /*inject_destination_collision=*/true)) << error;
        EXPECT_EQ(
            collision.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Error));
        EXPECT_EQ(
            collision.controller.error_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::InvalidCommand));
        EXPECT_EQ(collision.controller.current_durable_epoch, expected_epoch);
        // Opening a new snapshot preserves the last sealed publication for
        // delayed readers. Invalid policy must neither erase that receipt nor
        // publish the rejected transaction as a new transport command.
        EXPECT_EQ(collision.controller.transaction_id, expected_transaction + 1u);
        EXPECT_EQ(collision.controller.command_transaction, expected_transaction);
        EXPECT_EQ(collision.command.transaction_id, expected_transaction);
        EXPECT_EQ(collision.transport_groups_acquired, 0u);
        EXPECT_EQ(collision.transport_groups_prepared, 0u);
        EXPECT_EQ(collision.transport_groups_published, 0u);
        EXPECT_EQ(collision.transport_groups_retired, 0u);
        EXPECT_EQ(collision.transport_groups_restored, 0u);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         ContinuationSiblingsFreezeOneEpochAcrossAdversarialAdmissionPublication)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto policy_input = adversarialPolicyInput();
        for (const DeviceType continuation_type :
             {DeviceType::CUDA, DeviceType::ROCm})
        {
            SCOPED_TRACE(
                continuation_type == DeviceType::CUDA
                    ? "cuda_continuation"
                    : "rocm_continuation");
            const auto resolved_topology = topology(continuation_type);
            auto fabrics = makeControllerFabricPair(
                resolved_topology, policy_input);
            const auto &continuation_group = resolved_topology->groups.at(
                static_cast<std::size_t>(
                    resolved_topology->leader_group_id));
            const std::size_t expected_members =
                continuation_type == DeviceType::CUDA ? 2u : 4u;
            ASSERT_EQ(
                continuation_group.participant_ids.size(), expected_members);

            IBackend *const continuation_backend =
                continuation_type == DeviceType::CUDA ? cuda : rocm;
            const auto *const continuation_fabric =
                continuation_type == DeviceType::CUDA
                    ? fabrics.cuda_rank.get()
                    : fabrics.rocm_rank.get();
            ASSERT_NE(continuation_fabric, nullptr);
            const std::string initial_diagnostic =
                continuation_fabric->describeInferenceEpochBarrier();
            EXPECT_NE(
                initial_diagnostic.find("published_sequence=0"),
                std::string::npos);
            EXPECT_NE(
                initial_diagnostic.find("arrivals=["),
                std::string::npos);

            std::vector<std::unique_ptr<InferenceEpochBarrierEndpoint>>
                endpoints;
            endpoints.reserve(expected_members);
            InferenceEpochBarrierEndpoint *publisher = nullptr;
            for (const int participant_id :
                 continuation_group.participant_ids)
            {
                const auto binding =
                    continuation_fabric->participantBinding(participant_id);
                ASSERT_EQ(binding.device.type, continuation_type);
                ASSERT_TRUE(binding.inference_epoch_member);
                endpoints.push_back(std::make_unique<
                    InferenceEpochBarrierEndpoint>(
                    continuation_backend,
                    binding,
                    policy_input.base_epoch));
                if (binding.authority_leader)
                    publisher = endpoints.back().get();
            }
            ASSERT_NE(publisher, nullptr);

            /* Every local bank reaches E+1 while global admission remains E.
             * The publisher then arrives alone, reproducing the exact window in
             * which independent graph-root sampling used to diverge. */
            for (auto &endpoint : endpoints)
                ASSERT_TRUE(endpoint->publishNextLocalEpoch());
            ASSERT_TRUE(publisher->enqueueAcquire());

            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(5);
            std::uint64_t publisher_arrival = 0u;
            bool arrival_copy_ok = true;
            while (publisher_arrival != 1u &&
                   std::chrono::steady_clock::now() < deadline)
            {
                arrival_copy_ok =
                    publisher->copyArrival(&publisher_arrival);
                if (!arrival_copy_ok)
                    break;
                if (publisher_arrival != 1u)
                    std::this_thread::yield();
            }
            EXPECT_TRUE(arrival_copy_ok);
            EXPECT_EQ(publisher_arrival, 1u)
                << "publisher acquire did not raise its mapped arrival guard";

            const std::uint64_t admitted_epoch =
                policy_input.base_epoch + 1u;
            EXPECT_TRUE(publisher->publishGlobalAdmission(admitted_epoch));
            for (auto &endpoint : endpoints)
            {
                if (endpoint.get() != publisher)
                    ASSERT_TRUE(endpoint->enqueueAcquire());
            }
            for (auto &endpoint : endpoints)
                ASSERT_TRUE(endpoint->awaitInference());

            std::vector<DeviceMoEOverlayEpochTicket> tickets(
                endpoints.size());
            std::vector<DeviceMoEOverlayEpochStatus> statuses(
                endpoints.size());
            for (std::size_t index = 0u;
                 index < endpoints.size();
                 ++index)
            {
                ASSERT_TRUE(endpoints[index]->copyAcquireEvidence(
                    &tickets[index], &statuses[index]));
                ASSERT_TRUE(statuses[index].succeeded())
                    << "participant=" << endpoints[index]->participantId()
                    << " code=" << statuses[index].code;
                EXPECT_EQ(tickets[index].epoch, admitted_epoch);
                EXPECT_EQ(statuses[index].epoch, admitted_epoch);
                EXPECT_EQ(tickets[index].epoch, tickets.front().epoch);
            }

            MoEOverlayDeviceControllerInferenceEpochRecord observed_barrier{};
            ASSERT_TRUE(publisher->copyBarrier(&observed_barrier));
            EXPECT_EQ(observed_barrier.epoch, admitted_epoch);
            EXPECT_EQ(observed_barrier.publication_sequence, 1u);
            EXPECT_NE(
                continuation_fabric->describeInferenceEpochBarrier().find(
                    "published_sequence=1"),
                std::string::npos);
            EXPECT_EQ(
                observed_barrier.publisher_participant_id,
                static_cast<std::uint32_t>(
                    resolved_topology->leader_participant_id));
            for (const int participant_id :
                 continuation_group.participant_ids)
            {
                EXPECT_EQ(
                    observed_barrier.arrival_sequence[participant_id], 1u);
            }

            for (auto &endpoint : endpoints)
                ASSERT_TRUE(endpoint->enqueueRelease());
            for (auto &endpoint : endpoints)
                ASSERT_TRUE(endpoint->awaitInference());
            for (std::size_t index = 0u;
                 index < endpoints.size();
                 ++index)
            {
                ASSERT_TRUE(endpoints[index]->copyAcquireEvidence(
                    &tickets[index], &statuses[index]));
                ASSERT_TRUE(statuses[index].succeeded());
                EXPECT_EQ(tickets[index].epoch, 0u);
                EXPECT_EQ(tickets[index].selector, 0u);
            }
        }
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CapturedStaticAuthorityRunsOnEveryCudaAndRocmParticipantGraph)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        const auto cuda_context = context(0);
        const auto rocm_context = context(1);
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_fabric;
        std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_fabric;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;

        // The production constructor is a bilateral first-touch rendezvous;
        // retain that real concurrent setup rather than substituting host test
        // pages for either backend.
        std::thread cuda_builder([&]
                                 {
                                     try
                                     {
                                         cuda_fabric = std::make_shared<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             MoEOverlayNodeLocalDeviceControllerFabric::Config{
                                                 .mpi_ctx = cuda_context,
                                                 .topology = resolved_topology,
                                                 .num_layers = 2u,
                                                 .num_experts = 16u,
                                                 .command_capacity = 8u,
                                                 .initial_durable_epoch = 3u,
                                             });
                                     }
                                     catch (...)
                                     {
                                         cuda_error = std::current_exception();
                                     }
                                 });
        std::thread rocm_builder([&]
                                 {
                                     try
                                     {
                                         rocm_fabric = std::make_shared<
                                             MoEOverlayNodeLocalDeviceControllerFabric>(
                                             MoEOverlayNodeLocalDeviceControllerFabric::Config{
                                                 .mpi_ctx = rocm_context,
                                                 .topology = resolved_topology,
                                                 .num_layers = 2u,
                                                 .num_experts = 16u,
                                                 .command_capacity = 8u,
                                                 .initial_durable_epoch = 3u,
                                             });
                                     }
                                     catch (...)
                                     {
                                         rocm_error = std::current_exception();
                                     }
                                 });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);

        std::vector<std::unique_ptr<ControllerRuntimeFixture>>
            runtime_fixtures;
        std::vector<MoEOverlayDeviceControllerRuntimeBinding>
            cuda_runtime_bindings;
        std::vector<MoEOverlayDeviceControllerRuntimeBinding>
            rocm_runtime_bindings;
        runtime_fixtures.reserve(resolved_topology->participants.size());
        for (const auto &participant : resolved_topology->participants)
        {
            const auto *const group = resolved_topology
                                          ->groupForParticipant(
                                              participant.participant_id);
            ASSERT_NE(group, nullptr);
            IBackend *const backend =
                participant.device.type == DeviceType::CUDA ? cuda : rocm;
            auto fixture = std::make_unique<ControllerRuntimeFixture>(
                backend,
                participant.device,
                participant.participant_id,
                static_cast<std::uint32_t>(
                    participant.domain_participant_index),
                static_cast<std::uint32_t>(
                    group->participant_ids.size()));
            if (participant.world_rank == 0)
                cuda_runtime_bindings.push_back(fixture->binding());
            else
                rocm_runtime_bindings.push_back(fixture->binding());
            runtime_fixtures.push_back(std::move(fixture));
        }

        auto cuda_service = std::make_unique<
            MoEOverlayDeviceControllerGraphService>(
            MoEOverlayDeviceControllerGraphService::Config{
                .mpi_ctx = cuda_context,
                .topology = resolved_topology,
                .fabric = cuda_fabric,
                .runtime_bindings = cuda_runtime_bindings,
            });
        auto rocm_service = std::make_unique<
            MoEOverlayDeviceControllerGraphService>(
            MoEOverlayDeviceControllerGraphService::Config{
                .mpi_ctx = rocm_context,
                .topology = resolved_topology,
                .fabric = rocm_fabric,
                .runtime_bindings = rocm_runtime_bindings,
            });
        ASSERT_EQ(cuda_service->localGraphCount(), 2u);
        ASSERT_EQ(rocm_service->localGraphCount(), 4u);
        ASSERT_TRUE(cuda_service->ownsLeaderGraph());
        ASSERT_FALSE(rocm_service->ownsLeaderGraph());
        EXPECT_EQ(
            cuda_service->state(),
            MoEOverlayDeviceControllerActivationState::Prepared);
        EXPECT_EQ(
            rocm_service->state(),
            MoEOverlayDeviceControllerActivationState::Prepared);

        bool cuda_ok = false;
        bool rocm_ok = false;
        std::string cuda_message;
        std::string rocm_message;
        std::thread cuda_certifier([&]
                                   {
                                       cuda_ok = cuda_service
                                                     ->certifyStaticNoMovement(
                                                         &cuda_message);
                                   });
        std::thread rocm_certifier([&]
                                   {
                                       rocm_ok = rocm_service
                                                     ->certifyStaticNoMovement(
                                                         &rocm_message);
                                   });
        cuda_certifier.join();
        rocm_certifier.join();
        EXPECT_TRUE(cuda_ok) << cuda_message;
        EXPECT_TRUE(rocm_ok) << rocm_message;

        /*
         * Static certification is setup work, not implicit service
         * activation. Mirror the production post-capture edge explicitly and
         * prove both backend owners reject a second transition.
         */
        cuda_service->start();
        rocm_service->start();
        EXPECT_EQ(
            cuda_service->state(),
            MoEOverlayDeviceControllerActivationState::Running);
        EXPECT_EQ(
            rocm_service->state(),
            MoEOverlayDeviceControllerActivationState::Running);
        EXPECT_THROW(cuda_service->start(), std::logic_error);
        EXPECT_THROW(rocm_service->start(), std::logic_error);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         DeviceLocalServiceTotalsPublishAsCoherentMappedSnapshots)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 2 ||
            rocm->deviceCount() < 4)
        {
            GTEST_SKIP() << "Requires two CUDA and four ROCm devices";
        }

        const auto resolved_topology = topology();
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> cuda_fabric;
        std::unique_ptr<MoEOverlayNodeLocalDeviceControllerFabric> rocm_fabric;
        std::exception_ptr cuda_error;
        std::exception_ptr rocm_error;
        const auto make_config = [&](int rank)
        {
            return MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = context(rank),
                .topology = resolved_topology,
                .num_layers = 2u,
                .num_experts = 16u,
                .command_capacity = 8u,
                .initial_durable_epoch = 3u,
            };
        };
        std::thread cuda_builder([&]
        {
            try
            {
                cuda_fabric = std::make_unique<
                    MoEOverlayNodeLocalDeviceControllerFabric>(
                    make_config(0));
            }
            catch (...)
            {
                cuda_error = std::current_exception();
            }
        });
        std::thread rocm_builder([&]
        {
            try
            {
                rocm_fabric = std::make_unique<
                    MoEOverlayNodeLocalDeviceControllerFabric>(
                    make_config(1));
            }
            catch (...)
            {
                rocm_error = std::current_exception();
            }
        });
        cuda_builder.join();
        rocm_builder.join();
        if (cuda_error)
            std::rethrow_exception(cuda_error);
        if (rocm_error)
            std::rethrow_exception(rocm_error);

        const auto prove_backend = [resolved_topology](
            IBackend *backend,
            MoEOverlayNodeLocalDeviceControllerFabric &fabric,
            DeviceType type)
        {
            const auto participant = std::find_if(
                resolved_topology->participants.begin(),
                resolved_topology->participants.end(),
                [type](const auto &candidate)
                {
                    return candidate.device.type == type;
                });
            ASSERT_NE(participant, resolved_topology->participants.end());
            const auto binding = fabric.participantBinding(
                participant->participant_id);
            ASSERT_TRUE(binding.valid());

            constexpr std::size_t kLayers = 2u;
            constexpr std::size_t kCells =
                kLayers * kDeviceMoEOverlayServicePhaseCount;
            std::array<DeviceMoEOverlayServiceTelemetryCell, kCells> host{};
            std::array<DeviceMoEOverlayServiceTelemetrySample, kLayers>
                host_samples{};
            for (std::size_t cell = 0u; cell < host.size(); ++cell)
            {
                host[cell].total_nanoseconds = 100u + cell;
                host[cell].activation_count = 10u + cell;
                host[cell].sample_count = 1u + cell;
            }

            const int ordinal = binding.device.ordinal;
            auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(
                binding.device);
            auto *const device_cells = static_cast<
                DeviceMoEOverlayServiceTelemetryCell *>(backend->allocate(
                sizeof(host), ordinal));
            auto *const device_samples = static_cast<
                DeviceMoEOverlayServiceTelemetrySample *>(backend->allocate(
                sizeof(host_samples), ordinal));
            void *const stream = backend->createStream(ordinal);
            void *const terminal = backend->createEvent(ordinal);
            const auto cleanup = [&]
            {
                kernel.reset();
                if (terminal)
                    backend->destroyEvent(terminal, ordinal);
                if (stream)
                    backend->destroyStream(stream, ordinal);
                if (device_cells)
                    backend->free(device_cells, ordinal);
                if (device_samples)
                    backend->free(device_samples, ordinal);
            };
            ASSERT_TRUE(
                kernel && device_cells && device_samples && stream &&
                terminal);
            const MoEKernelLaunchContext launch{.stream = stream};
            ASSERT_TRUE(backend->hostToDevice(
                device_cells,
                host.data(),
                sizeof(host),
                ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDevice(
                device_samples,
                host_samples.data(),
                sizeof(host_samples),
                ordinal,
                stream));
            ASSERT_TRUE(kernel->publishMoEOverlayServiceTelemetry(
                launch,
                device_cells,
                device_samples,
                kLayers,
                participant->participant_id,
                binding.service_telemetry_publication));
            ASSERT_TRUE(backend->recordEvent(terminal, ordinal, stream));
            ASSERT_TRUE(await(
                backend, terminal, ordinal, std::chrono::seconds(5)));

            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            std::uint64_t generation = 0u;
            ASSERT_TRUE(fabric.trySnapshotServiceTelemetry(
                participant->participant_id, &rows, &generation));
            ASSERT_EQ(generation, 2u);
            ASSERT_EQ(rows.size(), kLayers);
            for (std::size_t layer = 0u; layer < kLayers; ++layer)
            {
                ASSERT_TRUE(rows[layer].valid());
                EXPECT_EQ(rows[layer].participant_id,
                          participant->participant_id);
                EXPECT_EQ(rows[layer].layer, static_cast<int>(layer));
                for (std::size_t phase = 0u;
                     phase < kDeviceMoEOverlayServicePhaseCount;
                     ++phase)
                {
                    const auto &cell = host[
                        layer * kDeviceMoEOverlayServicePhaseCount + phase];
                    EXPECT_EQ(rows[layer].total_nanoseconds[phase],
                              cell.total_nanoseconds);
                    EXPECT_EQ(rows[layer].activation_count[phase],
                              cell.activation_count);
                    EXPECT_EQ(rows[layer].sample_count[phase],
                              cell.sample_count);
                }
            }

            for (auto &cell : host)
            {
                cell.total_nanoseconds += 500u;
                cell.activation_count += 50u;
                cell.sample_count += 5u;
            }
            ASSERT_TRUE(backend->hostToDevice(
                device_cells,
                host.data(),
                sizeof(host),
                ordinal,
                stream));
            ASSERT_TRUE(kernel->publishMoEOverlayServiceTelemetry(
                launch,
                device_cells,
                device_samples,
                kLayers,
                participant->participant_id,
                binding.service_telemetry_publication));
            ASSERT_TRUE(backend->recordEvent(terminal, ordinal, stream));
            ASSERT_TRUE(await(
                backend, terminal, ordinal, std::chrono::seconds(5)));
            ASSERT_TRUE(fabric.trySnapshotServiceTelemetry(
                participant->participant_id, &rows, &generation));
            EXPECT_EQ(generation, 4u);
            EXPECT_EQ(rows[1].sample_count[2], host[5].sample_count);
            cleanup();
        };

        prove_backend(cuda, *cuda_fabric, DeviceType::CUDA);
        prove_backend(rocm, *rocm_fabric, DeviceType::ROCm);
    }

    TEST(Test__MoEOverlayDeviceControllerFabricCUDAAndROCm,
         CapturedServiceMarkersAccumulateOnCUDAAndROCm)
    {
        IBackend *const cuda = getCUDABackend();
        IBackend *const rocm = getROCmBackend();
        if (!cuda || !rocm || cuda->deviceCount() < 1 ||
            rocm->deviceCount() < 1)
        {
            GTEST_SKIP() << "Requires one CUDA and one ROCm device";
        }

        const auto prove_backend = [](IBackend *backend, DeviceId device)
        {
            constexpr std::uint32_t kExperts = 16u;
            constexpr std::uint64_t kPrefillActivations = 7u;
            DeviceMoELayerRuntime host_runtime{};
            host_runtime.expert_count = kExperts;
            host_runtime.top_k = 1u;
            host_runtime.prefill_local_histogram[0] =
                kPrefillActivations;
            std::array<DeviceMoEOverlayServiceTelemetryCell,
                       kDeviceMoEOverlayServicePhaseCount>
                host_cells{};
            DeviceMoEOverlayServiceTelemetrySample host_sample{};

            const int ordinal = device.ordinal;
            auto kernel =
                llaminar::v2::kernels::KernelFactory::createMoEKernel(
                    device);
            auto *const device_runtime = static_cast<DeviceMoELayerRuntime *>(
                backend->allocate(sizeof(host_runtime), ordinal));
            auto *const device_cells = static_cast<
                DeviceMoEOverlayServiceTelemetryCell *>(backend->allocate(
                sizeof(host_cells), ordinal));
            auto *const device_sample = static_cast<
                DeviceMoEOverlayServiceTelemetrySample *>(backend->allocate(
                sizeof(host_sample), ordinal));
            void *const stream = backend->createStream(ordinal);
            void *const terminal = backend->createEvent(ordinal);
            std::unique_ptr<IGPUGraphCapture> graph;
            const auto cleanup = [&]
            {
                graph.reset();
                kernel.reset();
                if (terminal)
                    backend->destroyEvent(terminal, ordinal);
                if (stream)
                    backend->destroyStream(stream, ordinal);
                if (device_sample)
                    backend->free(device_sample, ordinal);
                if (device_cells)
                    backend->free(device_cells, ordinal);
                if (device_runtime)
                    backend->free(device_runtime, ordinal);
            };

            ASSERT_TRUE(
                kernel && device_runtime && device_cells && device_sample &&
                stream && terminal);
            ASSERT_TRUE(backend->hostToDevice(
                device_runtime,
                &host_runtime,
                sizeof(host_runtime),
                ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDevice(
                device_cells,
                host_cells.data(),
                sizeof(host_cells),
                ordinal,
                stream));
            ASSERT_TRUE(backend->hostToDevice(
                device_sample,
                &host_sample,
                sizeof(host_sample),
                ordinal,
                stream));
            ASSERT_TRUE(backend->recordEvent(terminal, ordinal, stream));
            ASSERT_TRUE(await(
                backend, terminal, ordinal, std::chrono::seconds(5)));

            bool captured = false;
            std::vector<GPUGraphKernelNodeInfo> nodes;
            std::string inspection_error;
            auto &worker =
                GPUDeviceContextPool::instance().getContext(device);
            worker.submitAndWait(
                [&]
                {
                    graph = worker.createGraphCapture(stream);
                    const MoEKernelLaunchContext launch{.stream = stream};
                    captured = graph && graph->beginCapture() &&
                               kernel->beginMoEOverlayServiceTelemetry(
                                   launch, device_sample) &&
                               kernel->finishMoEOverlayServiceTelemetry(
                                   launch,
                                   device_runtime,
                                   device_cells,
                                   device_sample,
                                   kExperts,
                                   MoEOverlayServicePhaseHint::Prefill) &&
                               graph->endCapture() &&
                               graph->inspectKernelNodes(
                                   nodes, &inspection_error) &&
                               graph->instantiate() &&
                               graph->launchOnStream(stream) &&
                               backend->recordEvent(
                                   terminal, ordinal, stream);
                });
            ASSERT_TRUE(captured) << inspection_error;
            ASSERT_EQ(nodes.size(), 2u);
            ASSERT_TRUE(await(
                backend, terminal, ordinal, std::chrono::seconds(5)));

            ASSERT_TRUE(backend->deviceToHost(
                host_cells.data(),
                device_cells,
                sizeof(host_cells),
                ordinal,
                stream));
            ASSERT_TRUE(backend->deviceToHost(
                &host_sample,
                device_sample,
                sizeof(host_sample),
                ordinal,
                stream));
            ASSERT_TRUE(backend->recordEvent(terminal, ordinal, stream));
            ASSERT_TRUE(await(
                backend, terminal, ordinal, std::chrono::seconds(5)));

            const auto &prefill = host_cells[1];
            EXPECT_EQ(prefill.sample_count, 1u);
            EXPECT_EQ(prefill.activation_count, kPrefillActivations);
            EXPECT_GT(prefill.total_nanoseconds, 0u);
            EXPECT_EQ(prefill.dropped_samples, 0u);
            EXPECT_EQ(host_sample.armed, 0u);
            cleanup();
        };

        prove_backend(cuda, DeviceId::cuda(0));
        prove_backend(rocm, DeviceId::rocm(0));
    }
} // namespace llaminar2::test
