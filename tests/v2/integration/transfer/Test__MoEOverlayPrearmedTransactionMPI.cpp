/**
 * @file Test__MoEOverlayPrearmedTransactionMPI.cpp
 * @brief Two-process proof for pre-armed bounded ExpertOverlay device epochs.
 *
 * The production-shaped topology places two CUDA participants on MPI rank zero
 * and four ROCm participants on rank one. Every follower captures and launches
 * its complete zero-movement Dynamic transaction before the authority leader is
 * submitted. Movement uses a separate bounded retirement graph only after an
 * authenticated device reader receipt; no controller graph remains resident
 * across physical preparation or an inference grace period. Replaying the
 * retained graphs proves that transaction monotonicity and teardown do not
 * need the historical 17 host-decided phases.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "execution/moe/MoEOverlayDevicePlacementPolicy.h"
#include "execution/moe/MoEOverlayDeviceTransportProtocol.h"
#include "execution/moe/MoEOverlayEconomyProfileComposer.h"
#include "execution/moe/MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "execution/moe/MoEOverlayDeviceControllerRuntimeBinding.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "utils/MPIContext.h"

#include "MoEOverlayDeviceRuntimePublicationFixture.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        inline constexpr std::uint32_t kLayers = 2u;
        inline constexpr std::uint32_t kExperts = 16u;
        inline constexpr std::uint32_t kTopK = 2u;
        inline constexpr std::uint64_t kInitialEpoch = 3u;
        inline constexpr std::uint32_t kReplayCount = 3u;

        /** @return Backend authority for one exact GPU participant. */
        IBackend *backendFor(DeviceId device) noexcept
        {
            if (device.is_cuda())
                return getCUDABackend();
            if (device.is_rocm())
                return getROCmBackend();
            return nullptr;
        }

        /** Turn one rank-local predicate into a matched MPI checkpoint. */
        bool allRanksSucceeded(bool local_success) noexcept
        {
            int local = local_success ? 1 : 0;
            int global = 0;
            return MPI_Allreduce(
                       &local,
                       &global,
                       1,
                       MPI_INT,
                       MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS &&
                   global != 0;
        }

        /** Poll one exact terminal event without synchronizing its stream. */
        bool awaitEvent(
            IBackend *backend,
            DeviceId device,
            void *event,
            std::chrono::steady_clock::duration timeout) noexcept
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            while (backend && event &&
                   std::chrono::steady_clock::now() < deadline)
            {
                bool ready = false;
                if (!backend->queryEvent(
                        event, device.gpu_ordinal(), &ready))
                {
                    return false;
                }
                if (ready)
                    return true;
                std::this_thread::yield();
            }
            return false;
        }

        /** Build one homogeneous rank-local routed-expert domain. */
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

        /** @return The production 2xCUDA/4xROCm two-priority topology. */
        std::shared_ptr<const MoEOverlayDeviceControllerTopology>
        makeTopology()
        {
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
                domain("continuation", DeviceType::CUDA, 2, 0),
                domain("secondary", DeviceType::ROCm, 4, 1),
            };
            plan.routed_tiers = {
                {
                    .name = "priority0",
                    .domain = "continuation",
                    .priority = 0,
                },
                {
                    .name = "priority1",
                    .domain = "secondary",
                    .priority = 1,
                    .fallback = true,
                },
            };
            std::vector<int> tiers(kExperts, 1);
            std::fill_n(tiers.begin(), 4u, 0);
            for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
            {
                plan.placements.push_back({
                    .layer = static_cast<int>(layer),
                    .routed_expert_tier = tiers,
                });
            }
            auto resolved = resolveMoEOverlayDeviceControllerTopology(
                plan,
                MoEExpertOwnerMap::build(plan),
                {.world_rank_node_ids = std::array<int, 2>{0, 0}});
            return std::make_shared<const MoEOverlayDeviceControllerTopology>(
                std::move(resolved));
        }

        /**
         * Build an exact, valid owner snapshot with no routing observations.
         *
         * Zero demand intentionally selects Dynamic no movement regardless of
         * tier service prices. The topology remains non-trivial: four experts
         * reside in the continuation tier and twelve in the secondary tier.
         */
        MoEOverlayDevicePlacementPolicyInput makeZeroDemandInput(
            const MoEOverlayDeviceControllerTopology &topology)
        {
            MoEOverlayDevicePlacementPolicyInput input;
            input.num_layers = kLayers;
            input.num_experts = kExperts;
            input.base_epoch = kInitialEpoch;
            input.minimum_window_activations = 1u;
            input.maximum_cycles_per_wave = 2u;
            input.command_capacity = 16u;
            input.dynamic_maximum_commands_per_wave = 16u;
            input.payload_bytes_per_layer.assign(kLayers, 4096u);
            input.participants.reserve(topology.participants.size());
            for (const auto &participant : topology.participants)
            {
                const auto *const group =
                    topology.groupForParticipant(participant.participant_id);
                if (!group)
                {
                    throw std::logic_error(
                        "pre-armed topology participant has no controller group");
                }
                input.participants.push_back({
                    .participant_id = static_cast<std::uint32_t>(
                        participant.participant_id),
                    .tier_priority = group->tier_priority,
                    .tier_index = participant.tier_idx,
                });
            }
            input.collected_state.assign(
                topology.participants.size() * kLayers * kExperts,
                0u);
            for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
            {
                for (std::uint32_t expert = 0u; expert < kExperts; ++expert)
                {
                    const std::uint32_t owner = expert < 4u
                        ? expert % 2u
                        : 2u + (expert - 4u) % 4u;
                    const std::size_t offset =
                        (static_cast<std::size_t>(owner) * kLayers + layer) *
                            kExperts +
                        expert;
                    input.collected_state[offset] =
                        moe_rebalance_policy::packCollectedState(
                            /*activation_count=*/0u,
                            /*active_transfer_slots=*/0u,
                            /*physically_resident=*/true,
                            /*transfer_backed=*/false,
                            /*authoritative_owner=*/true);
                }
            }

            // Device Dynamic policy has one mandatory, fully measured economy
            // contract even when this transaction has no routed demand. Keep
            // every service price equal so the proof isolates lifecycle rather
            // than accidentally authoring a promotion or skew cycle.
            MoEOverlayDevicePlacementEconomyInput economy;
            economy.tier_count = 2u;
            economy.routed_experts_per_token = kTopK;
            economy.transaction_generation = 1u;
            const std::size_t plane_words =
                static_cast<std::size_t>(kLayers) * kExperts;
            economy.phase_expert_demand.assign(
                kMoEOverlayDeviceControllerDemandPhaseCount * plane_words,
                0u);
            economy.service_costs.assign(
                static_cast<std::size_t>(economy.tier_count) * kLayers *
                    kMoEOverlayDeviceControllerEconomyServicePhaseCount,
                10u);
            economy.migration_costs.assign(
                topology.participants.size() * topology.participants.size() *
                    kLayers,
                {});
            for (std::uint32_t source = 0u;
                 source < topology.participants.size(); ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < topology.participants.size(); ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                    {
                        economy.migration_costs[
                            (static_cast<std::size_t>(source) *
                                 topology.participants.size() +
                             destination) *
                                kLayers +
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
            if (!input.valid())
            {
                throw std::logic_error(
                    "pre-armed transaction fixture produced an invalid owner snapshot");
            }
            return input;
        }

        /** Build a capacity-preserving but deliberately adversarial hotness map. */
        MoEOverlayDevicePlacementPolicyInput makeAdversarialDemandInput(
            const MoEOverlayDeviceControllerTopology &topology)
        {
            auto input = makeZeroDemandInput(topology);
            const std::array<std::uint64_t, kExperts> demand{
                10u, 20u, 30u, 40u,
                1600u, 1500u, 1400u, 1300u,
                1200u, 1100u, 1000u, 900u,
                800u, 700u, 600u, 500u,
            };
            for (std::uint32_t participant = 0u;
                 participant < topology.participants.size(); ++participant)
            {
                for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                {
                    for (std::uint32_t expert = 0u; expert < kExperts; ++expert)
                    {
                        const std::size_t offset =
                            (static_cast<std::size_t>(participant) * kLayers +
                             layer) *
                                kExperts +
                            expert;
                        if (!moe_rebalance_policy::
                                collectedStateAuthoritativeOwner(
                                    input.collected_state[offset]))
                        {
                            continue;
                        }
                        input.collected_state[offset] =
                            moe_rebalance_policy::packCollectedState(
                                demand[expert],
                                /*active_transfer_slots=*/0u,
                                /*physically_resident=*/true,
                                /*transfer_backed=*/false,
                                /*authoritative_owner=*/true);
                    }
                }
            }
            auto &economy = *input.economy;
            for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
            {
                for (std::uint32_t expert = 0u; expert < kExperts; ++expert)
                {
                    economy.phase_expert_demand[
                        (static_cast<std::size_t>(
                             kMoEOverlayDeviceControllerEconomyDecodePhase) *
                             kLayers +
                         layer) *
                            kExperts +
                        expert] = demand[expert];
                }
            }
            for (std::uint32_t tier = 0u; tier < economy.tier_count; ++tier)
            {
                for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                {
                    for (std::uint32_t phase = 0u;
                         phase <
                         kMoEOverlayDeviceControllerEconomyServicePhaseCount;
                         ++phase)
                    {
                        economy.service_costs[
                            (static_cast<std::size_t>(tier) * kLayers + layer) *
                                kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                            phase] = tier == 0u ? 10u : 100u;
                    }
                }
            }
            if (!input.valid())
                throw std::logic_error("adversarial pre-armed input is invalid");
            return input;
        }

        /** Publish deterministic, complete cost matrices from the sole leader. */
        void publishEconomy(
            MoEOverlayNodeLocalDeviceControllerFabric &fabric,
            const MoEOverlayDeviceControllerTopology &topology)
        {
            if (!fabric.ownsEconomyPublication())
                return;
            auto service = std::make_shared<MoERoutedTierServiceProfile>();
            service->identity = "prearmed-transaction-service-v1";
            service->active_sources = {true, true, true};
            for (std::uint32_t tier = 0u; tier < 2u; ++tier)
            {
                for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                {
                    service->costs.push_back({
                        .tier_index = static_cast<int>(tier),
                        .layer = static_cast<int>(layer),
                        .nanoseconds_per_activation = tier == 0u
                            ? std::array<std::uint64_t, 3>{10u, 10u, 10u}
                            : std::array<std::uint64_t, 3>{100u, 100u, 100u},
                    });
                }
            }
            auto migration =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            migration->identity = "prearmed-transaction-migration-v1";
            for (std::uint32_t source = 0u;
                 source < topology.participants.size(); ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < topology.participants.size(); ++destination)
                {
                    if (source == destination)
                        continue;
                    for (std::uint32_t layer = 0u; layer < kLayers; ++layer)
                    {
                        migration->costs.push_back({
                            .source_participant = static_cast<int>(source),
                            .destination_participant =
                                static_cast<int>(destination),
                            .layer = static_cast<int>(layer),
                            .transfer_and_repack_ns = 1u,
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
            if (!profiles.valid())
                throw std::logic_error("pre-armed economy profile is invalid");
            fabric.publishCertifiedEconomyProfiles(profiles);
        }

        /**
         * @brief Own one complete retained transaction for a local participant.
         *
         * Followers may be launched before the leader. Their first mapped wait
         * is contained inside the graph, so graph submission cannot race an
         * authority ticket and no host phase scheduler is involved.
         */
        class PrearmedEndpoint final
        {
        public:
            PrearmedEndpoint(
                MoEOverlayDeviceControllerParticipantBinding binding,
                MoEOverlayDeviceRuntimePublicationFixture *fixture,
                MoEOverlayDeviceDemandPhase demand_phase,
                bool split_retirement = false,
                bool direct_empty_decision = false)
                : binding_(std::move(binding)),
                  fixture_(fixture),
                  backend_(backendFor(binding_.device)),
                  demand_phase_(demand_phase),
                  split_retirement_(split_retirement),
                  direct_empty_decision_(direct_empty_decision)
            {
                if (!binding_.valid() || !fixture_ || !backend_)
                {
                    throw std::invalid_argument(
                        "pre-armed endpoint requires complete device resources");
                }
                auto &context = GPUDeviceContextPool::instance().getContext(
                    binding_.device);
                kernel_ = llaminar::v2::kernels::KernelFactory::createMoEKernel(
                    binding_.device);
                stream_ = backend_->createStream(binding_.device.ordinal);
                terminal_ = backend_->createEvent(binding_.device.ordinal);
                retirement_terminal_ = backend_->createEvent(
                    binding_.device.ordinal);
                reader_stream_ = backend_->createStream(binding_.device.ordinal);
                reader_event_ = backend_->createEvent(binding_.device.ordinal);
                if (binding_.authority_leader)
                {
                    policy_ = static_cast<
                        MoEOverlayDeviceControllerPolicyResult *>(
                        backend_->allocate(
                            sizeof(MoEOverlayDeviceControllerPolicyResult),
                            binding_.device.ordinal));
                }
                if (!kernel_ || !stream_ || !terminal_ ||
                    !retirement_terminal_ || !reader_stream_ ||
                    !reader_event_ ||
                    (binding_.authority_leader && !policy_))
                {
                    throw std::runtime_error(
                        "pre-armed endpoint allocation failed");
                }

                const auto &runtime = fixture_->runtimeBinding();
                const DeviceMoERebalanceConfig snapshot_config{
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
                };
                if (!validateDeviceMoERebalanceConfig(snapshot_config))
                    throw std::logic_error("invalid pre-armed snapshot geometry");

                graph_ = context.createGraphCapture(stream_);
                if (!graph_ || !graph_->beginCapture())
                    throw std::runtime_error("could not begin pre-armed capture");
                bool capture_open = true;
                try
                {
                    const MoEKernelLaunchContext launch{.stream = stream_};
                    const auto action = [&](
                                            MoEOverlayDeviceControllerAction op,
                                            MoEOverlayDeviceControllerTransactionKind kind =
                                                MoEOverlayDeviceControllerTransactionKind::Invalid,
                                            MoEOverlayDeviceDemandPhase phase =
                                                MoEOverlayDeviceDemandPhase::Invalid)
                    {
                        const bool policy_action =
                            op == MoEOverlayDeviceControllerAction::
                                      AuthorDynamicPolicy ||
                            op == MoEOverlayDeviceControllerAction::
                                      PublishCommand;
                        const bool publication_action =
                            op == MoEOverlayDeviceControllerAction::
                                      ApplyRuntimeCandidate ||
                            op == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeCandidate ||
                            op == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeRetirement;
                        const bool readiness_action =
                            op == MoEOverlayDeviceControllerAction::
                                      PublishRuntimeRetirementReadiness;
                        return kernel_->runMoEOverlayDeviceControllerAction(
                            launch,
                            {
                                .binding = binding_.deviceBinding(),
                                .action = op,
                                .transaction_kind = kind,
                                .demand_phase = phase,
                                .policy_result = policy_action
                                                     ? policy_
                                                     : nullptr,
                                .runtime_publication = publication_action
                                    ? fixture_->publicationBinding()
                                    : MoEOverlayDeviceRuntimePublicationBinding{},
                                .retirement_readiness = readiness_action
                                    ? MoEOverlayDeviceRetirementReadinessBinding{
                                          .epoch_control = fixture_
                                              ->runtimeBinding()
                                              .epoch_control,
                                      }
                                    : MoEOverlayDeviceRetirementReadinessBinding{},
                            });
                    };

                    bool captured = true;
                    if (binding_.authority_leader)
                    {
                        captured = action(
                            MoEOverlayDeviceControllerAction::BeginTransaction,
                            MoEOverlayDeviceControllerTransactionKind::
                                DynamicPlacement,
                            demand_phase_);
                    }
                    captured = captured &&
                        kernel_->packDeviceRebalanceHistograms(
                            launch,
                            runtime.runtime_layers_device,
                            binding_.participant_collected_state,
                            snapshot_config) &&
                        action(
                            MoEOverlayDeviceControllerAction::
                                PublishParticipantSnapshot);
                    if (binding_.group_root)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::
                                PublishGroupSnapshot);
                    }
                    if (binding_.authority_leader)
                    {
                        captured = captured &&
                            action(
                                MoEOverlayDeviceControllerAction::
                                    AuthorDynamicPolicy,
                                MoEOverlayDeviceControllerTransactionKind::
                                    Invalid,
                                demand_phase_) &&
                            action(
                                MoEOverlayDeviceControllerAction::
                                    PublishCommand);
                    }
                    if (direct_empty_decision_)
                    {
                        if (binding_.authority_leader)
                        {
                            captured = captured && action(
                                MoEOverlayDeviceControllerAction::
                                    CompleteEmptyDynamicDecision);
                        }
                    }
                    else
                    {
                    // Runtime nodes are unconditional in the retained graph.
                    // The device validates and elides them only after policy
                    // publishes an authenticated empty Dynamic command.
                    captured = captured && action(
                        MoEOverlayDeviceControllerAction::
                            ApplyRuntimeCandidate);
                    if (binding_.group_root)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::
                                AcknowledgePrepared);
                    }
                    if (binding_.authority_leader)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::BeginCommit);
                    }
                    captured = captured && action(
                        MoEOverlayDeviceControllerAction::
                            PublishRuntimeCandidate);
                    if (binding_.group_root)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::
                                AcknowledgePublished);
                    }
                    if (binding_.authority_leader)
                    {
                        captured = captured &&
                            action(
                                MoEOverlayDeviceControllerAction::
                                    PublishAdmission) &&
                            action(
                                MoEOverlayDeviceControllerAction::
                                    BeginDynamicRetirement);
                    }
                    if (split_retirement_)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirementReadiness);
                    }
                    if (!split_retirement_)
                    {
                        captured = captured && action(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirement);
                        if (binding_.group_root)
                        {
                            captured = captured && action(
                                MoEOverlayDeviceControllerAction::
                                    AcknowledgeRetired);
                        }
                        captured = captured && action(
                            binding_.authority_leader
                                ? MoEOverlayDeviceControllerAction::
                                      CompleteDynamicRetirement
                                : MoEOverlayDeviceControllerAction::
                                      AwaitTransactionComplete);
                    }
                    }
                    if (!captured || !graph_->endCapture() ||
                        !graph_->instantiate())
                    {
                        throw std::runtime_error(
                            "complete pre-armed transaction capture failed");
                    }
                    capture_open = false;

                    if (split_retirement_)
                    {
                        retirement_graph_ = context.createGraphCapture(stream_);
                        if (!retirement_graph_ ||
                            !retirement_graph_->beginCapture())
                        {
                            throw std::runtime_error(
                                "could not begin bounded retirement capture");
                        }
                        capture_open = true;
                        bool retirement_captured = action(
                            MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirement);
                        if (binding_.group_root)
                        {
                            retirement_captured = retirement_captured &&
                                action(
                                    MoEOverlayDeviceControllerAction::
                                        AcknowledgeRetired);
                        }
                        retirement_captured = retirement_captured && action(
                            binding_.authority_leader
                                ? MoEOverlayDeviceControllerAction::
                                      CompleteDynamicRetirement
                                : MoEOverlayDeviceControllerAction::
                                      AwaitTransactionComplete);
                        if (!retirement_captured ||
                            !retirement_graph_->endCapture() ||
                            !retirement_graph_->instantiate())
                        {
                            throw std::runtime_error(
                                "bounded retirement capture failed");
                        }
                        capture_open = false;
                    }
                }
                catch (...)
                {
                    if (capture_open)
                        (void)graph_->endCapture();
                    throw;
                }
            }

            /** Destroy the graph before releasing its embedded addresses. */
            ~PrearmedEndpoint()
            {
                retirement_graph_.reset();
                graph_.reset();
                kernel_.reset();
                if (policy_)
                    backend_->free(policy_, binding_.device.ordinal);
                if (terminal_)
                    backend_->destroyEvent(terminal_, binding_.device.ordinal);
                if (retirement_terminal_)
                    backend_->destroyEvent(
                        retirement_terminal_, binding_.device.ordinal);
                if (reader_event_)
                    backend_->destroyEvent(
                        reader_event_, binding_.device.ordinal);
                if (stream_)
                    backend_->destroyStream(stream_, binding_.device.ordinal);
                if (reader_stream_)
                    backend_->destroyStream(
                        reader_stream_, binding_.device.ordinal);
            }

            PrearmedEndpoint(const PrearmedEndpoint &) = delete;
            PrearmedEndpoint &operator=(const PrearmedEndpoint &) = delete;

            /** Submit one replay and publish its exact terminal event. */
            bool launch() noexcept
            {
                return graph_ && graph_->launchOnStream(stream_) &&
                       backend_->recordEvent(
                           terminal_, binding_.device.ordinal, stream_);
            }

            /** Poll the exact terminal for one replay. */
            bool await() noexcept
            {
                return awaitEvent(
                    backend_,
                    binding_.device,
                    terminal_,
                    std::chrono::seconds(10));
            }

            /** Submit the bounded reader-drained retirement epoch. */
            bool launchRetirement() noexcept
            {
                return split_retirement_ && retirement_graph_ &&
                       retirement_graph_->launchOnStream(stream_) &&
                       backend_->recordEvent(
                           retirement_terminal_,
                           binding_.device.ordinal,
                           stream_);
            }

            /** Poll the exact bounded-retirement terminal. */
            bool awaitRetirement() noexcept
            {
                return awaitEvent(
                    backend_,
                    binding_.device,
                    retirement_terminal_,
                    std::chrono::seconds(10));
            }

            /** Acquire one immutable runtime epoch on an independent stream. */
            bool acquireReader(std::uint32_t slot) noexcept
            {
                return kernel_->acquireMoEOverlayEpoch(
                           {.stream = reader_stream_},
                           fixture_->runtimeBinding().epoch_control,
                           fixture_->requestTicket(slot),
                           fixture_->requestStatus(slot),
                           &binding_.controller->admission_epoch) &&
                    backend_->recordEvent(
                        reader_event_,
                        binding_.device.ordinal,
                        reader_stream_);
            }

            /** Release one prior acquire without entering the controller stream. */
            bool releaseReader(std::uint32_t slot) noexcept
            {
                return kernel_->releaseMoEOverlayEpoch(
                           {.stream = reader_stream_},
                           fixture_->runtimeBinding().epoch_control,
                           fixture_->requestTicket(slot),
                           fixture_->requestStatus(slot)) &&
                    kernel_->runMoEOverlayDeviceControllerAction(
                        {.stream = reader_stream_},
                        {
                            .binding = binding_.deviceBinding(),
                            .action = MoEOverlayDeviceControllerAction::
                                PublishRuntimeRetirementReadiness,
                            .retirement_readiness = {
                                .epoch_control = fixture_->runtimeBinding()
                                                     .epoch_control,
                            },
                        }) &&
                    backend_->recordEvent(
                        reader_event_,
                        binding_.device.ordinal,
                        reader_stream_);
            }

            /** Poll the latest independent inference-reader terminal. */
            bool awaitReader() noexcept
            {
                return awaitEvent(
                    backend_,
                    binding_.device,
                    reader_event_,
                    std::chrono::seconds(10));
            }

            /** Copy the semantic completion of one reader operation. */
            bool copyReaderStatus(
                std::uint32_t slot,
                DeviceMoEOverlayEpochStatus *status) noexcept
            {
                return fixture_->copyRequestStatus(slot, status);
            }

            /** @return Whether this endpoint owns the sole authority graph. */
            bool authorityLeader() const noexcept
            {
                return binding_.authority_leader;
            }

            /** @return Immutable global participant id for diagnostics. */
            int participantId() const noexcept
            {
                return binding_.participant_id;
            }

        private:
            MoEOverlayDeviceControllerParticipantBinding binding_;
            MoEOverlayDeviceRuntimePublicationFixture *fixture_ = nullptr;
            IBackend *backend_ = nullptr;
            MoEOverlayDeviceDemandPhase demand_phase_ =
                MoEOverlayDeviceDemandPhase::Invalid;
            bool split_retirement_ = false;
            bool direct_empty_decision_ = false;
            std::unique_ptr<IMoEKernel> kernel_;
            void *stream_ = nullptr;
            void *terminal_ = nullptr;
            void *retirement_terminal_ = nullptr;
            void *reader_stream_ = nullptr;
            void *reader_event_ = nullptr;
            MoEOverlayDeviceControllerPolicyResult *policy_ = nullptr;
            std::unique_ptr<IGPUGraphCapture> graph_;
            std::unique_ptr<IGPUGraphCapture> retirement_graph_;
        };
    } // namespace

    TEST(Test__MoEOverlayPrearmedTransactionMPI,
         CompleteFollowerGraphsReplayWithoutHostPhaseScheduling)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
        if (world_size != 2)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        IBackend *const local_backend = world_rank == 0
            ? getCUDABackend()
            : getROCmBackend();
        const int required_devices = world_rank == 0 ? 2 : 4;
        const bool hardware_ready = local_backend &&
            local_backend->deviceCount() >= required_devices;
        if (!allRanksSucceeded(hardware_ready))
        {
            GTEST_SKIP() << "requires two CUDA devices on rank 0 and four ROCm devices on rank 1";
        }

        const auto topology = makeTopology();
        const auto policy_input = makeZeroDemandInput(*topology);
        auto mpi_context = std::make_shared<MPIContext>(
            world_rank, world_size, MPI_COMM_WORLD);
        auto fabric = std::make_shared<
            MoEOverlayNodeLocalDeviceControllerFabric>(
            MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = mpi_context,
                .topology = topology,
                .num_layers = kLayers,
                .num_experts = kExperts,
                .routed_experts_per_token = kTopK,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = kInitialEpoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .minimum_window_activations = 1u,
                .maximum_cycles_per_wave = 2u,
            });
        publishEconomy(*fabric, *topology);
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        ASSERT_TRUE(fabric->economyProfilesPublished());

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        std::vector<std::unique_ptr<PrearmedEndpoint>> endpoints;
        for (const int participant_id : fabric->localParticipantIds())
        {
            const auto &participant = topology->participants.at(
                static_cast<std::size_t>(participant_id));
            const auto binding = fabric->participantBinding(participant_id);
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                local_backend,
                participant,
                *topology,
                policy_input,
                &binding.controller->admission_epoch,
                binding.lifetime));
            endpoints.push_back(std::make_unique<PrearmedEndpoint>(
                binding,
                fixtures.back().get(),
                MoEOverlayDeviceDemandPhase::Prefill,
                /*split_retirement=*/false,
                /*direct_empty_decision=*/true));
        }

        for (std::uint32_t replay = 1u; replay <= kReplayCount; ++replay)
        {
            bool submitted = true;
            for (auto &endpoint : endpoints)
            {
                if (!endpoint->authorityLeader())
                    submitted = endpoint->launch() && submitted;
            }
            ASSERT_TRUE(allRanksSucceeded(submitted));

            // This is a setup/maintenance rendezvous, not an inference edge.
            // It proves every follower graph is already resident before the
            // sole leader can create transaction `replay`.
            ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
            if (world_rank == topology->leader_world_rank)
            {
                const auto leader = std::find_if(
                    endpoints.begin(),
                    endpoints.end(),
                    [](const auto &endpoint)
                    {
                        return endpoint->authorityLeader();
                    });
                ASSERT_NE(leader, endpoints.end());
                submitted = (*leader)->launch() && submitted;
            }
            ASSERT_TRUE(allRanksSucceeded(submitted));

            bool terminals_ready = true;
            for (auto &endpoint : endpoints)
            {
                const bool ready = endpoint->await();
                if (!ready)
                {
                    std::cerr << "prearmed terminal timeout rank=" << world_rank
                              << " replay=" << replay
                              << " participant=" << endpoint->participantId()
                              << '\n';
                }
                terminals_ready = ready && terminals_ready;
            }
            if (!terminals_ready)
            {
                const auto group = std::find_if(
                    topology->groups.begin(),
                    topology->groups.end(),
                    [world_rank](const auto &candidate)
                    {
                        return candidate.root_world_rank == world_rank;
                    });
                if (group != topology->groups.end())
                {
                    MoEOverlayDeviceTransportProtocol diagnostic(
                        fabric->transportBinding(group->group_id));
                    std::cerr << "prearmed lifecycle rank=" << world_rank
                              << ' ' << diagnostic.describeLifecycle()
                              << '\n';
                }
            }
            ASSERT_TRUE(allRanksSucceeded(terminals_ready));

            const auto local_group = std::find_if(
                topology->groups.begin(),
                topology->groups.end(),
                [world_rank](const auto &group)
                {
                    return group.root_world_rank == world_rank;
                });
            ASSERT_NE(local_group, topology->groups.end());
            const auto transport = fabric->transportBinding(
                local_group->group_id);
            ASSERT_TRUE(transport.valid());
            const auto state = std::atomic_ref<const std::uint32_t>(
                transport.controller->state)
                                   .load(std::memory_order_acquire);
            const auto transaction = std::atomic_ref<const std::uint64_t>(
                transport.controller->transaction_id)
                                         .load(std::memory_order_acquire);
            const auto durable_epoch = std::atomic_ref<const std::uint64_t>(
                transport.controller->current_durable_epoch)
                                         .load(std::memory_order_acquire);
            const auto command_count = std::atomic_ref<const std::uint32_t>(
                transport.command->command_count)
                                           .load(std::memory_order_acquire);
            EXPECT_EQ(
                state,
                static_cast<std::uint32_t>(
                    MoEOverlayDeviceControllerState::Complete));
            EXPECT_EQ(transaction, replay);
            EXPECT_EQ(durable_epoch, kInitialEpoch);
            EXPECT_EQ(command_count, 0u);
            ASSERT_TRUE(allRanksSucceeded(
                state == static_cast<std::uint32_t>(
                             MoEOverlayDeviceControllerState::Complete) &&
                transaction == replay &&
                durable_epoch == kInitialEpoch && command_count == 0u));
        }
    }

    TEST(Test__MoEOverlayPrearmedTransactionMPI,
         CompleteMovementGraphConsumesOnlyPhysicalReceipts)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);
        if (world_size != 2)
            GTEST_SKIP() << "requires exactly two MPI ranks";

        IBackend *const local_backend = world_rank == 0
            ? getCUDABackend()
            : getROCmBackend();
        const int required_devices = world_rank == 0 ? 2 : 4;
        if (!allRanksSucceeded(
                local_backend &&
                local_backend->deviceCount() >= required_devices))
        {
            GTEST_SKIP() << "requires two CUDA and four ROCm devices";
        }

        const auto topology = makeTopology();
        const auto policy_input = makeAdversarialDemandInput(*topology);
        const auto expected =
            MoEOverlayDevicePlacementPolicyReference::planDynamic(
                policy_input);
        ASSERT_TRUE(expected.hasMovement());
        ASSERT_GT(expected.evidence.promotions, 0u);
        ASSERT_GT(expected.evidence.demotions, 0u);

        auto mpi_context = std::make_shared<MPIContext>(
            world_rank, world_size, MPI_COMM_WORLD);
        auto fabric = std::make_shared<
            MoEOverlayNodeLocalDeviceControllerFabric>(
            MoEOverlayNodeLocalDeviceControllerFabric::Config{
                .mpi_ctx = mpi_context,
                .topology = topology,
                .num_layers = kLayers,
                .num_experts = kExperts,
                .routed_experts_per_token = kTopK,
                .command_capacity = policy_input.command_capacity,
                .initial_durable_epoch = kInitialEpoch,
                .payload_bytes_per_layer =
                    policy_input.payload_bytes_per_layer,
                .minimum_window_activations = 1u,
                .maximum_cycles_per_wave = 2u,
            });
        publishEconomy(*fabric, *topology);
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        ASSERT_TRUE(fabric->economyProfilesPublished());

        const auto local_group = std::find_if(
            topology->groups.begin(),
            topology->groups.end(),
            [world_rank](const auto &group)
            {
                return group.root_world_rank == world_rank;
            });
        ASSERT_NE(local_group, topology->groups.end());
        MoEOverlayDeviceTransportProtocol transport(
            fabric->transportBinding(local_group->group_id));

        std::vector<std::unique_ptr<
            MoEOverlayDeviceRuntimePublicationFixture>> fixtures;
        std::vector<std::unique_ptr<PrearmedEndpoint>> endpoints;
        for (const int participant_id : fabric->localParticipantIds())
        {
            const auto &participant = topology->participants.at(
                static_cast<std::size_t>(participant_id));
            const auto binding = fabric->participantBinding(participant_id);
            fixtures.push_back(std::make_unique<
                MoEOverlayDeviceRuntimePublicationFixture>(
                local_backend,
                participant,
                *topology,
                policy_input,
                &binding.controller->admission_epoch,
                binding.lifetime));
            endpoints.push_back(std::make_unique<PrearmedEndpoint>(
                binding,
                fixtures.back().get(),
                MoEOverlayDeviceDemandPhase::Decode,
                /*split_retirement=*/true));
        }

        // Hold one request on the base bank before maintenance starts. The
        // complete graph must later wait for this ticket without preventing an
        // independent request from using the newly published bank.
        bool old_readers_acquired = true;
        for (auto &endpoint : endpoints)
            old_readers_acquired = endpoint->acquireReader(0u) &&
                old_readers_acquired;
        for (auto &endpoint : endpoints)
            old_readers_acquired = endpoint->awaitReader() &&
                old_readers_acquired;
        ASSERT_TRUE(allRanksSucceeded(old_readers_acquired));

        bool submitted = true;
        for (auto &endpoint : endpoints)
        {
            if (!endpoint->authorityLeader())
                submitted = endpoint->launch() && submitted;
        }
        ASSERT_TRUE(allRanksSucceeded(submitted));
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        if (world_rank == topology->leader_world_rank)
        {
            const auto leader = std::find_if(
                endpoints.begin(),
                endpoints.end(),
                [](const auto &endpoint)
                {
                    return endpoint->authorityLeader();
                });
            ASSERT_NE(leader, endpoints.end());
            submitted = (*leader)->launch() && submitted;
        }
        ASSERT_TRUE(allRanksSucceeded(submitted));

        std::optional<MoEOverlayDeviceTransportCommandBatch> command;
        std::string local_error;
        const auto acquire_deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(10);
        while (!command &&
               std::chrono::steady_clock::now() < acquire_deadline)
        {
            auto acquired = transport.tryAcquire(/*after_transaction=*/0u);
            if (acquired.status ==
                MoEOverlayDeviceTransportAcquireStatus::Failed)
            {
                local_error = std::move(acquired.error);
                break;
            }
            if (acquired.status ==
                MoEOverlayDeviceTransportAcquireStatus::Ready)
            {
                command = std::move(acquired.batch);
                break;
            }
            std::this_thread::yield();
        }
        const bool acquired_movement = command && command->movesWeights() &&
            command->header.command_count > 0u;
        ASSERT_TRUE(allRanksSucceeded(acquired_movement)) << local_error;

        bool arrivals_ready = true;
        for (auto &fixture : fixtures)
        {
            arrivals_ready = fixture->enqueuePreparedArrivals(
                                 *command, &local_error) &&
                arrivals_ready;
        }
        for (auto &fixture : fixtures)
        {
            arrivals_ready = fixture->awaitPreparedArrivals(&local_error) &&
                arrivals_ready;
        }
        arrivals_ready = arrivals_ready &&
            transport.publishPrepared(*command, &local_error);
        ASSERT_TRUE(allRanksSucceeded(arrivals_ready)) << local_error;

        const auto wait_for = [](auto &&predicate)
        {
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (predicate())
                    return true;
                std::this_thread::yield();
            }
            return false;
        };
        bool published = wait_for(
            [&] { return transport.publicationReady(*command); });
        published = published &&
            transport.publishPublished(*command, &local_error);
        ASSERT_TRUE(allRanksSucceeded(published)) << local_error;

        const bool retirement_open = wait_for(
            [&] { return transport.retirementOpen(*command); });
        ASSERT_TRUE(allRanksSucceeded(retirement_open));

        bool publication_terminals_ready = true;
        for (auto &endpoint : endpoints)
        {
            publication_terminals_ready = endpoint->await() &&
                publication_terminals_ready;
        }
        ASSERT_TRUE(allRanksSucceeded(publication_terminals_ready));

        // This request must acquire/release E+1 while the controller block is
        // still waiting for the held E reader. Success proves maintenance did
        // not serialize the participant's inference stream.
        bool new_readers_completed = true;
        for (auto &endpoint : endpoints)
        {
            new_readers_completed = endpoint->acquireReader(1u) &&
                endpoint->releaseReader(1u) && new_readers_completed;
        }
        for (auto &endpoint : endpoints)
            new_readers_completed = endpoint->awaitReader() &&
                new_readers_completed;
        for (auto &endpoint : endpoints)
        {
            DeviceMoEOverlayEpochStatus status{};
            new_readers_completed = endpoint->copyReaderStatus(1u, &status) &&
                status.operation == static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochOperation::Release) &&
                status.code == static_cast<std::uint32_t>(
                    DeviceMoEOverlayEpochStatusCode::Success) &&
                new_readers_completed;
        }
        ASSERT_TRUE(allRanksSucceeded(new_readers_completed));

        bool old_readers_released = true;
        for (auto &endpoint : endpoints)
            old_readers_released = endpoint->releaseReader(0u) &&
                old_readers_released;
        for (auto &endpoint : endpoints)
            old_readers_released = endpoint->awaitReader() &&
                old_readers_released;
        ASSERT_TRUE(allRanksSucceeded(old_readers_released));

        const bool reader_ticket_ready = wait_for(
            [&] { return transport.runtimeReadersReady(*command); });
        ASSERT_TRUE(allRanksSucceeded(reader_ticket_ready));

        bool retirement_submitted = true;
        for (auto &endpoint : endpoints)
        {
            retirement_submitted = endpoint->launchRetirement() &&
                retirement_submitted;
        }
        ASSERT_TRUE(allRanksSucceeded(retirement_submitted));

        bool retired = wait_for(
            [&] { return transport.retirementRequested(*command); });
        retired = retired && transport.publishRetired(*command, &local_error);
        ASSERT_TRUE(allRanksSucceeded(retired)) << local_error;

        bool terminals_ready = true;
        for (auto &endpoint : endpoints)
        {
            terminals_ready = endpoint->awaitRetirement() &&
                terminals_ready;
        }
        ASSERT_TRUE(allRanksSucceeded(terminals_ready));

        const auto binding = fabric->transportBinding(local_group->group_id);
        const auto state = std::atomic_ref<const std::uint32_t>(
            binding.controller->state)
                               .load(std::memory_order_acquire);
        const auto durable_epoch = std::atomic_ref<const std::uint64_t>(
            binding.controller->current_durable_epoch)
                                    .load(std::memory_order_acquire);
        EXPECT_EQ(
            state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Complete));
        EXPECT_EQ(durable_epoch, kInitialEpoch + 1u);
        EXPECT_EQ(command->header.transaction_id, 1u);
        EXPECT_GT(command->header.command_count, 0u);
        ASSERT_TRUE(allRanksSucceeded(
            state == static_cast<std::uint32_t>(
                         MoEOverlayDeviceControllerState::Complete) &&
            durable_epoch == kInitialEpoch + 1u));
    }
} // namespace llaminar2::test
