/**
 * @file Test__MoEOverlayResidencyMaintenanceService.cpp
 * @brief Device-free adversarial tests for asynchronous overlay maintenance.
 *
 * The production worker must leave ticket admission open while a migration is
 * staged, preserve one frozen transaction through arbitrary backpressure, and
 * prove static immobility without ever invoking the physical transport. These
 * tests control each event edge independently so accidental synchronous or
 * inference-thread progress cannot make them pass.
 */

#include "execution/moe/MoEOverlayResidencyMaintenanceService.h"
#include "execution/moe/MoEOverlayMPIResidencyProposalPublisher.h"
#include "utils/ObservedExpertDemandFixture.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace std::chrono_literals;

        /** @brief Build one single-participant routed-expert domain. */
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
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Build one priority-ordered tier over a named domain. */
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

        /** @brief Return a hot/warm/cold plan suitable for CPU-only protocol tests. */
        MoERoutedExpertPlacementPlan threeTierPlan(
            RoutedExpertResidencyPolicy policy)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = policy;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                domain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
                domain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(1, 0),
                    1,
                    CollectiveBackendType::RCCL),
                domain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(2),
                    2,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "cuda_hot", 0, 2),
                tier("warm", "rocm_warm", 1, 2),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /** @brief Small real placement geometry with six routed experts. */
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

        /**
         * @brief Create a full window that forces promotions and demotions.
         *
         * Initial residency is hot={0,1}, warm={2,3}, cold={4,5}; the supplied
         * counts make expert 4 hottest and expert 1 cold enough to demote.
         */
        std::shared_ptr<DecodeExpertHistogram> fullMovementHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 6;
            config.top_k = 1;
            config.window_size = 370;
            config.token_boundary_layer_idx = 0;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                3,
                {0, 0, 1, 1, 2, 2});
            // One 370-row initial prefill, then at most one 400-row reversed
            // prefill. The controlled observer may retain the first window
            // while a proposal and its replacement briefly coexist.
            admitObservedExpertDemand(config, {370, 400, 1}, 3);
            auto histogram =
                std::make_shared<DecodeExpertHistogram>(std::move(config));
            const std::vector<uint64_t> counts{90, 20, 80, 10, 100, 70};
            recordObservedPrefillBatch(*histogram, 0, counts);
            return histogram;
        }

        /**
         * @brief Complete measured-policy fixture for protocol-only movement tests.
         *
         * The authority prices the parallel critical path using exact
         * participant rows; tier aggregates alone are only sufficient for
         * initial capacity placement.  This fixture therefore describes both
         * views of the same three single-participant tiers.
         */
        std::shared_ptr<const MoERoutedTierServiceProfile> serviceProfile()
        {
            auto profile = std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "maintenance-service-profile-v1";
            profile->production_topology =
                ExpertHistogramProductionTopology::uniform(
                    1, kAllExpertHistogramProductionSources);
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

        /** @brief Complete cheap directed migration profile for three endpoints. */
        std::shared_ptr<const MoEOverlayMigrationCostProfile>
        migrationProfile()
        {
            auto profile =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            profile->identity = "maintenance-migration-profile-v1";
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
                        .transfer_and_repack_ns = 1,
                        .inference_interference_ns = 0,
                    });
                }
            }
            return profile;
        }

        /** @brief Construct a shared dynamic publication authority. */
        std::shared_ptr<MoEOverlayResidencyAuthority> dynamicAuthority(
            DecodeExpertHistogram *histogram)
        {
            return std::make_shared<MoEOverlayResidencyAuthority>(
                MoEOverlayResidencyAuthority::Config{
                    .initial_plan = threeTierPlan(
                        RoutedExpertResidencyPolicy::RoutedTierRebalanced),
                    .model_metadata = modelMetadata(),
                    .maintenance_mode =
                        MoERebalanceRuntimeMode::Dynamic,
                    .histogram = histogram,
                    .phase_service_profile = serviceProfile(),
                    .migration_cost_profile = migrationProfile(),
                    .migration_economy_policy =
                        MoEOverlayMigrationEconomyPolicy{
                            .historical_window_weight = 0,
                            .current_window_weight = 1,
                            .payoff_horizon_tokens = 4,
                            .minimum_net_benefit_ns = 0,
                            .minimum_residency_generations = 0,
                        },
                    .perf_device = "device-free-test",
                });
        }

        /**
         * @brief Wait for a lock-free observable without sleeping a test thread.
         * @return Whether the predicate became true before the diagnostic bound.
         */
        template <typename Predicate>
        bool waitUntil(Predicate &&predicate, std::chrono::seconds bound = 5s)
        {
            const auto deadline = std::chrono::steady_clock::now() + bound;
            while (!predicate())
            {
                if (std::chrono::steady_clock::now() >= deadline)
                    return false;
                std::this_thread::yield();
            }
            return true;
        }

        /**
         * @brief Event-controlled transport that records transaction identity.
         *
         * Every readiness flag models an exact device/network event. Polls only
         * inspect flags, allowing the tests to prove that ticket execution can
         * continue while staging, preparation, or publication is pending.
         */
        class ControlledTransport final
            : public IMoEOverlayResidencyTransport
        {
        public:
            /** @brief One event-polled wave owned by the authority. */
            class Wave final : public IMoEOverlayResidencyWave
            {
            public:
                /** @brief Bind wave callbacks to the retained transport owner. */
                explicit Wave(ControlledTransport *owner) : owner_(owner) {}

                /** @brief Poll the test-controlled staging event. */
                MoEOverlayResidencyWaveProgress pollStage(
                    std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->stage_polls_;
                    owner_->cv_.notify_all();
                    if (owner_->defer_during_stage_)
                    {
                        return MoEOverlayResidencyWaveProgress::Deferred;
                    }
                    return owner_->stage_ready_
                               ? MoEOverlayResidencyWaveProgress::Ready
                               : MoEOverlayResidencyWaveProgress::Pending;
                }

                /** @brief Record inactive-bank preparation enqueue. */
                bool beginPrepare(std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->prepare_begins_;
                    owner_->cv_.notify_all();
                    return true;
                }

                /** @brief Poll the independently controlled prepare event. */
                MoEOverlayResidencyWaveProgress pollPrepare(
                    std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->prepare_polls_;
                    owner_->cv_.notify_all();
                    return owner_->prepare_ready_
                               ? MoEOverlayResidencyWaveProgress::Ready
                               : MoEOverlayResidencyWaveProgress::Pending;
                }

                /** @brief Record inference-visible selector publication. */
                bool beginPublication(std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->publication_begins_;
                    owner_->cv_.notify_all();
                    return true;
                }

                /** @brief Poll the independently controlled publication event. */
                MoEOverlayResidencyWaveProgress pollPublication(
                    std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->publication_polls_;
                    owner_->cv_.notify_all();
                    return owner_->publication_ready_
                               ? MoEOverlayResidencyWaveProgress::Ready
                               : MoEOverlayResidencyWaveProgress::Pending;
                }

                /** @brief Record unpublished cleanup ownership. */
                void abortStaged() noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->aborts_;
                    owner_->cv_.notify_all();
                }

                /** @brief Record lease-safe retirement on the worker. */
                void retirePrevious() noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->retirements_;
                    owner_->cv_.notify_all();
                }

            private:
                ControlledTransport *owner_ = nullptr;
            };

            /** @brief Snapshot of synchronized transport observations. */
            struct Observations
            {
                int begin_calls = 0;
                int waves_started = 0;
                int stage_polls = 0;
                int prepare_begins = 0;
                int prepare_polls = 0;
                int publication_begins = 0;
                int publication_polls = 0;
                int aborts = 0;
                int retirements = 0;
                uint64_t first_generation = 0;
                const DecodeExpertHistogramWindow *first_window = nullptr;
                bool retained_identity = true;
            };

            /**
             * @brief Start, defer, or fail a wave without waiting.
             * @param transaction Exact immutable proposal being retried.
             */
            MoEOverlayResidencyStageStart beginStage(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++begin_calls_;
                const auto *window = transaction.histogram_window.get();
                if (begin_calls_ == 1)
                {
                    first_generation_ = transaction.histogram_generation;
                    first_window_ = window;
                }
                else if (first_generation_ != transaction.histogram_generation ||
                         first_window_ != window)
                {
                    retained_identity_ = false;
                }
                cv_.notify_all();

                if (fail_start_)
                {
                    return {
                        .status =
                            MoEOverlayResidencyStageStartStatus::Failed,
                        .error = "injected physical transport failure",
                    };
                }
                if (!start_enabled_)
                {
                    return {
                        .status =
                            MoEOverlayResidencyStageStartStatus::Deferred,
                        .error = "injected destination backpressure",
                    };
                }

                ++waves_started_;
                return {
                    .status = MoEOverlayResidencyStageStartStatus::Started,
                    .wave = std::make_unique<Wave>(this),
                };
            }

            /** @brief Enable or defer destination-slot reservation. */
            void setStartEnabled(bool enabled)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                start_enabled_ = enabled;
                cv_.notify_all();
            }

            /** @brief Enable the exact staging-completion event. */
            void setStageReady(bool ready)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stage_ready_ = ready;
                cv_.notify_all();
            }

            /** @brief Inject global-style backpressure after a wave has started. */
            void setDeferDuringStage(bool deferred)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                defer_during_stage_ = deferred;
                cv_.notify_all();
            }

            /** @brief Enable the exact inactive-bank preparation event. */
            void setPrepareReady(bool ready)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                prepare_ready_ = ready;
                cv_.notify_all();
            }

            /** @brief Enable the exact selector-publication event. */
            void setPublicationReady(bool ready)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                publication_ready_ = ready;
                cv_.notify_all();
            }

            /** @brief Make the next physical stage request fail fatally. */
            void setFailStart(bool fail)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                fail_start_ = fail;
                cv_.notify_all();
            }

            /** @return Synchronized copy of all protocol observations. */
            Observations observations() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return {
                    .begin_calls = begin_calls_,
                    .waves_started = waves_started_,
                    .stage_polls = stage_polls_,
                    .prepare_begins = prepare_begins_,
                    .prepare_polls = prepare_polls_,
                    .publication_begins = publication_begins_,
                    .publication_polls = publication_polls_,
                    .aborts = aborts_,
                    .retirements = retirements_,
                    .first_generation = first_generation_,
                    .first_window = first_window_,
                    .retained_identity = retained_identity_,
                };
            }

        private:
            friend class Wave;
            mutable std::mutex mutex_;
            std::condition_variable cv_;
            bool start_enabled_ = true;
            bool defer_during_stage_ = false;
            bool stage_ready_ = false;
            bool prepare_ready_ = false;
            bool publication_ready_ = false;
            bool fail_start_ = false;
            int begin_calls_ = 0;
            int waves_started_ = 0;
            int stage_polls_ = 0;
            int prepare_begins_ = 0;
            int prepare_polls_ = 0;
            int publication_begins_ = 0;
            int publication_polls_ = 0;
            int aborts_ = 0;
            int retirements_ = 0;
            uint64_t first_generation_ = 0;
            const DecodeExpertHistogramWindow *first_window_ = nullptr;
            bool retained_identity_ = true;
        };

        /** @brief Immediate thread-safe canonical-plan publisher for CPU tests. */
        class ImmediateProposalPublisher final
            : public IMoEOverlayResidencyProposalPublisher
        {
        public:
            /**
             * @brief Construct a coordinator or an initially armed peer.
             * @param coordinator Whether this process owns routing evidence.
             * @param peer_proposal First authenticated peer plan, when any.
             */
            ImmediateProposalPublisher(
                bool coordinator,
                std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                    peer_proposal = {})
                : coordinator_(coordinator),
                  peer_proposal_(std::move(peer_proposal)),
                  active_(!coordinator)
            {
            }

            /** @brief Capture the exact coordinator plan and begin its send. */
            bool beginPublish(
                const MoEOverlayDistributedResidencyProposal &proposal,
                std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!coordinator_ || active_ || !proposal.valid())
                {
                    if (error)
                        *error = "invalid immediate proposal publication";
                    return false;
                }
                published_proposal_ =
                    std::make_shared<
                        MoEOverlayDistributedResidencyProposal>(proposal);
                active_ = true;
                ++publication_begins_;
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Complete a send or deliver one copied peer proposal. */
            MoEOverlayResidencyWaveProgress poll(
                std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                    *received,
                std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++polls_;
                if (received)
                    received->reset();
                if (!active_)
                {
                    if (error)
                        *error = "immediate proposal lane is idle";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (coordinator_)
                {
                    if (!publication_ready_)
                        return MoEOverlayResidencyWaveProgress::Pending;
                    active_ = false;
                    ++publication_completions_;
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (acknowledgement_pending_)
                {
                    acknowledgement_pending_ = false;
                    active_ = false;
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (!peer_proposal_)
                    return MoEOverlayResidencyWaveProgress::Pending;

                awaiting_generation_ =
                    peer_proposal_->plan.histogram_window->generation;
                if (received)
                    *received = std::move(peer_proposal_);
                else
                    peer_proposal_.reset();
                active_ = false;
                awaiting_validation_ = true;
                ++deliveries_;
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Complete device-free semantic adoption before readiness. */
            bool acceptReceivedProposal(
                std::uint64_t histogram_generation,
                std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (coordinator_ || !awaiting_validation_ ||
                    histogram_generation != awaiting_generation_)
                {
                    if (error)
                        *error = "invalid immediate proposal acceptance";
                    return false;
                }
                awaiting_validation_ = false;
                awaiting_generation_ = 0u;
                acknowledgement_pending_ = true;
                active_ = true;
                ++acceptances_;
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Retain one adversarial semantic rejection for inspection. */
            void rejectReceivedProposal(
                std::uint64_t histogram_generation,
                std::string diagnostic) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                rejected_generation_ = histogram_generation;
                rejection_ = std::move(diagnostic);
                awaiting_validation_ = false;
                acknowledgement_pending_ = false;
                active_ = false;
                ++rejections_;
            }

            /** @brief Arm the peer for another generation immediately. */
            bool armReceive(std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (coordinator_ || active_ || awaiting_validation_ ||
                    acknowledgement_pending_)
                {
                    if (error)
                        *error = "invalid immediate proposal rearm";
                    return false;
                }
                active_ = true;
                ++receive_rearms_;
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Mark the device-free lane stopped. */
            void stopAndDrain() override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                active_ = false;
                awaiting_validation_ = false;
                acknowledgement_pending_ = false;
            }

            /** @return Whether this fixture is the coordinator. */
            [[nodiscard]] bool isCoordinator() const noexcept override
            {
                return coordinator_;
            }

            /** @return Number of coordinator publication starts. */
            int publicationBegins() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return publication_begins_;
            }

            /** @return Number of completed coordinator publications. */
            int publicationCompletions() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return publication_completions_;
            }

            /** @return Number of non-blocking lane progress polls. */
            int polls() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return polls_;
            }

            /** @return Number of authenticated peer proposal deliveries. */
            int deliveries() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return deliveries_;
            }

            /** @return Number of immediate next-generation peer rearms. */
            int receiveRearms() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return receive_rearms_;
            }

            /** @brief Release or retain the coordinator publication terminal. */
            void setPublicationReady(bool ready)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                publication_ready_ = ready;
            }

            /** @return Coordinator proposal copied at beginPublish(). */
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
            publishedProposal() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return published_proposal_;
            }

        private:
            bool coordinator_ = false;
            mutable std::mutex mutex_;
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                peer_proposal_;
            std::shared_ptr<const MoEOverlayDistributedResidencyProposal>
                published_proposal_;
            bool active_ = false;
            bool awaiting_validation_ = false;
            bool acknowledgement_pending_ = false;
            bool publication_ready_ = true;
            std::uint64_t awaiting_generation_ = 0u;
            std::uint64_t rejected_generation_ = 0u;
            std::string rejection_;
            int publication_begins_ = 0;
            int publication_completions_ = 0;
            int polls_ = 0;
            int deliveries_ = 0;
            int acceptances_ = 0;
            int rejections_ = 0;
            int receive_rearms_ = 0;
        };

        /** @brief Start a service with a fast, non-semantic test cadence. */
        std::unique_ptr<MoEOverlayResidencyMaintenanceService> startService(
            std::shared_ptr<MoEOverlayResidencyAuthority> authority,
            std::shared_ptr<ControlledTransport> transport,
            std::shared_ptr<IMoEOverlayResidencyProposalPublisher>
                publisher = {})
        {
            auto service = std::make_unique<
                MoEOverlayResidencyMaintenanceService>(
                MoEOverlayResidencyMaintenanceService::Config{
                    .authority = std::move(authority),
                    .transport = std::move(transport),
                    .proposal_publisher = std::move(publisher),
                    .idle_poll_interval = 100us,
                    .perf_device = "device-free-test",
                });
            service->start();
            return service;
        }
    } // namespace

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        ConstructionRemainsPreparedUntilCompositionConsensusStartsWorker)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        MoEOverlayResidencyMaintenanceService service({
            .authority = authority,
            .transport = transport,
            .idle_poll_interval = 100us,
            .perf_device = "prepared-lifecycle-test",
        });

        std::this_thread::sleep_for(2ms);
        EXPECT_EQ(service.state(), MoEOverlayMaintenanceState::Prepared);
        EXPECT_EQ(service.stats().worker_starts, 0u);
        service.start();
        ASSERT_TRUE(waitUntil(
            [&] { return service.stats().worker_starts == 1u; }));
        EXPECT_THROW(service.start(), std::logic_error);
        service.stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        EXPECT_EQ(service.state(), MoEOverlayMaintenanceState::Stopped);
        EXPECT_THROW(service.start(), std::logic_error);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        PreparedServiceStopsWithoutCreatingAWorker)
    {
        auto histogram = fullMovementHistogram();
        MoEOverlayResidencyMaintenanceService service({
            .authority = dynamicAuthority(histogram.get()),
            .transport = std::make_shared<ControlledTransport>(),
            .idle_poll_interval = 100us,
            .perf_device = "prepared-stop-test",
        });

        ASSERT_EQ(service.state(), MoEOverlayMaintenanceState::Prepared);
        service.stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        service.stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        EXPECT_EQ(service.state(), MoEOverlayMaintenanceState::Stopped);
        EXPECT_EQ(service.stats().worker_starts, 0u);
        EXPECT_THROW(service.start(), std::logic_error);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DistributedReadinessRoundPausesButDoesNotStopGPUServicePublication)
    {
        using Action = MoEOverlayDeviceServicePublicationAction;
        using State = MoEOverlayEconomyCertificationState;

        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(
                State::CalibratingMovement),
            Action::Pause);
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(
                State::AwaitingServiceEvidence),
            Action::PollAndImport);
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(
                State::ExchangingServiceReadiness),
            Action::Pause)
            << "An unsuccessful all-rank readiness round must be resumable";
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(
                State::ExchangingServiceEvidence),
            Action::Stop)
            << "The ready snapshot is immutable once evidence exchange begins";
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(
                State::RebasingRoutingEvidence),
            Action::Stop)
            << "Rebase owns the immutable profile and needs no new service rows";
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(State::Complete),
            Action::Stop);
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(State::Failed),
            Action::Stop);
        EXPECT_EQ(
            moeOverlayDeviceServicePublicationAction(State::Stopped),
            Action::Stop);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        PublicActivityDistinguishesDemandReconciliationFromSettledCollection)
    {
        using Activity = MoEOptimizationActivityState;
        using State = MoEOverlayMaintenanceState;

        EXPECT_EQ(
            moeOptimizationActivityState(State::Waiting),
            Activity::CollectingDemand);
        EXPECT_EQ(
            moeOptimizationActivityState(State::ReconcilingDemand),
            Activity::ReconcilingDemand);
        EXPECT_EQ(
            moeOptimizationActivityState(State::DrainingEvidence),
            Activity::ReconcilingDemand);
        EXPECT_EQ(
            moeOptimizationActivityState(State::PlanningProposal),
            Activity::PlanningMovement);
        EXPECT_EQ(
            moeOptimizationActivityState(State::PublishingProposal),
            Activity::ExchangingProposal);
        EXPECT_EQ(
            moeOptimizationActivityState(State::ReceivingProposal),
            Activity::AwaitingAuthorityProposal);
        EXPECT_EQ(
            moeOptimizationActivityState(State::Staging),
            Activity::MovingWeights);
        EXPECT_EQ(
            moeOptimizationActivityState(State::Preparing),
            Activity::MovingWeights);
        EXPECT_EQ(
            moeOptimizationActivityState(State::Publishing),
            Activity::PublishingResidency);

        const MoEOptimizationStatus settled{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = Activity::CollectingDemand,
        };
        const MoEOptimizationStatus reconciling{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = Activity::ReconcilingDemand,
        };
        const MoEOptimizationStatus notified_but_not_reconciled{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = Activity::CollectingDemand,
            .published_progress_generation = 8u,
            .reconciled_progress_generation = 7u,
        };
        const MoEOptimizationStatus notification_reconciled{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = Activity::CollectingDemand,
            .demand_window = {
                .generation = 4u,
                .collected_routed_rows = 87u,
                .capacity_routed_rows = 640u,
            },
            .published_progress_generation = 8u,
            .reconciled_progress_generation = 8u,
        };
        EXPECT_TRUE(settled.quiescentBetweenWaves());
        EXPECT_FALSE(reconciling.quiescentBetweenWaves())
            << "A completed publication is not a timing boundary until the "
               "authority proves that no full demand window is queued";
        EXPECT_FALSE(notified_but_not_reconciled.quiescentBetweenWaves())
            << "A stale Waiting activity cannot hide inference progress that "
               "the policy worker has not reconciled";
        EXPECT_TRUE(notification_reconciled.quiescentBetweenWaves());
        EXPECT_TRUE(notification_reconciled.canBeginExclusiveCohort(552u))
            << "The complete cohort should fit strictly below the next decision";

        auto exact_boundary = notification_reconciled;
        exact_boundary.demand_window.collected_routed_rows = 88u;
        EXPECT_FALSE(exact_boundary.canBeginExclusiveCohort(552u))
            << "Completing the histogram on the cohort's final row would admit "
               "an asynchronous publication before terminal validation";

        auto no_authoritative_window = notification_reconciled;
        no_authoritative_window.demand_window = {};
        EXPECT_FALSE(no_authoritative_window.canBeginExclusiveCohort(1u));
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        LogicalInferenceCadenceReconcilesDeviceOnlyDemandAndPublishesMovement)
    {
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 6;
        histogram_config.top_k = 2;
        histogram_config.window_size = 4;
        histogram_config.token_boundary_layer_idx = 0;
        histogram_config.sockets = {
            DeviceId::cuda(0),
            DeviceId::rocm(0),
            DeviceId::cpu(),
        };
        histogram_config.ownership = MoELayeredExpertOwnership::uniform(
            1,
            3,
            {0, 0, 1, 1, 2, 2});
        admitObservedExpertDemand(histogram_config, {4, 4, 2}, 3);
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            std::move(histogram_config));

        std::atomic<int> drain_polls{0};
        histogram->registerRuntimeHistogramDrain(
            [&]()
            {
                const int poll = drain_polls.fetch_add(
                                     1, std::memory_order_relaxed) +
                                 1;
                if (poll == 1)
                    return RuntimeExpertHistogramDrainResult::pending();
                // The completed drain retains one real four-row verifier
                // invocation, not unrelated counts attached to four tokens.
                const int device_routes[]{0, 4, 2, 4, 0, 5, 4, 5};
                recordObservedExpertBatch(*histogram, 0,
                    ExpertHistogramSource::GroupedVerifier, device_routes);
                return RuntimeExpertHistogramDrainResult::ready();
            });

        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            { return service->optimizationStatus().quiescentBetweenWaves(); }));
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            {
                const auto status = service->optimizationStatus();
                return status.quiescentBetweenWaves() &&
                       status.published_progress_generation ==
                           status.reconciled_progress_generation;
            }));
        EXPECT_EQ(drain_polls.load(std::memory_order_relaxed), 0)
            << "A wake without inference cadence must not poll device banks";

        service->notifyInferenceProgress({
            .phase = MoEOverlayInferenceProgressPhase::Decode,
            .completed_logical_tokens = 3u,
        });
        ASSERT_TRUE(waitUntil(
            [&]
            {
                const auto status = service->optimizationStatus();
                return status.quiescentBetweenWaves() &&
                       status.published_progress_generation ==
                           status.reconciled_progress_generation;
            }));
        EXPECT_EQ(drain_polls.load(std::memory_order_relaxed), 0)
            << "The cadence must retain a partial window without eager D2H";

        service->notifyInferenceProgress({
            .phase = MoEOverlayInferenceProgressPhase::Decode,
            .completed_logical_tokens = 1u,
        });
        ASSERT_TRUE(waitUntil(
            [&] { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&] { return service->stats().committed_waves == 1u; }))
            << "Publication and its typed completion ledger must both settle";
        EXPECT_EQ(drain_polls.load(std::memory_order_relaxed), 2);
        EXPECT_EQ(service->stats().inference_progress_tokens, 4u);
        EXPECT_EQ(service->stats().histogram_runtime_probes, 1u);
        EXPECT_EQ(service->stats().histogram_runtime_reconciliations, 0u)
            << "The first reconciled device bank was already proposal-ready";
        EXPECT_EQ(service->stats().committed_waves, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        TypedEconomyProofRequiresExactAuthorityArithmetic)
    {
        const MoEOptimizationMovementEconomy valid{
            .authority = MoEOptimizationAuthority::Host,
            .transaction = 2u,
            .candidate_epoch = 2u,
            .command_count = 4u,
            .cycle_count = 2u,
            .proof = MoEOptimizationTimeEconomy{
            .projected_service_gain_ns = 1'000u,
            .projected_transfer_and_repack_ns = 200u,
            .projected_inference_interference_ns = 100u,
            .projected_net_benefit_ns = 700u,
            },
        };
        EXPECT_TRUE(valid.valid());

        auto mismatched_net = valid;
        std::get<MoEOptimizationTimeEconomy>(mismatched_net.proof).projected_net_benefit_ns = 701u;
        EXPECT_FALSE(mismatched_net.valid());

        auto follower_fabrication = valid;
        follower_fabrication.authority = MoEOptimizationAuthority::None;
        EXPECT_FALSE(follower_fabrication.valid());

        auto overflowed_cost = valid;
        std::get<MoEOptimizationTimeEconomy>(overflowed_cost.proof).projected_transfer_and_repack_ns =
            std::numeric_limits<std::uint64_t>::max();
        EXPECT_FALSE(overflowed_cost.valid());
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        TypedHostAdmissionProofRequiresTotalClassificationAndExactCapacity)
    {
        const MoEOptimizationHostMovementAdmission valid{
            .authority = MoEOptimizationAuthority::Host,
            .transaction = 2u,
            .candidate_epoch = 2u,
            .cycle_capacity_kind =
                MoEOptimizationCycleCapacityKind::Bounded,
            .maximum_concurrent_cycles = 2u,
            .candidate_cycles = 4u,
            .policy_eligible_cycles = 3u,
            .policy_eligible_axes = {
                .tier_residency = 1u,
                .participant_placement = 1u,
                .combined = 1u,
            },
            .admitted_candidate_cycles = 2u,
            .admitted_candidate_axes = {
                .tier_residency = 1u,
                .participant_placement = 1u,
            },
            .admitted_physical_cycles = 2u,
            .admitted_physical_axes = {
                .tier_residency = 1u,
                .combined = 1u,
            },
            .individual_policy_rejected_cycles = 1u,
            .dependent_payoff_rejected_cycles = 1u,
            .physical_cycle_recomposition = false,
            .capacity_bounded = false,
            .policy_bounded = true,
        };
        EXPECT_TRUE(valid.valid());

        auto follower_fabrication = valid;
        follower_fabrication.authority = MoEOptimizationAuthority::None;
        EXPECT_FALSE(follower_fabrication.valid());

        auto unclassified_candidate = valid;
        ++unclassified_candidate.candidate_cycles;
        EXPECT_FALSE(unclassified_candidate.valid());

        auto axis_double_count = valid;
        ++axis_double_count.admitted_physical_axes.combined;
        EXPECT_FALSE(axis_double_count.valid());

        auto over_capacity = valid;
        over_capacity.maximum_concurrent_cycles = 1u;
        EXPECT_FALSE(over_capacity.valid());

        auto mislabeled_capacity = valid;
        mislabeled_capacity.capacity_bounded = true;
        EXPECT_FALSE(mislabeled_capacity.valid());

        /* Dependency cohorts are combinatorial economy-search alternatives,
         * not candidate cycles. The production 122B stress run exposed a
         * proof that had three eligible cycles, admitted two, rejected one
         * remaining cycle, and separately rejected one cohort. Counting that
         * cohort as a second cycle made an otherwise exact proof invalid. */
        auto with_rejected_cohort = valid;
        with_rejected_cohort.dependent_cohort_candidates = 1u;
        with_rejected_cohort.dependent_cohort_payoff_rejections = 1u;
        EXPECT_TRUE(with_rejected_cohort.valid());

        auto cohort_mislabeled_as_cycle = with_rejected_cohort;
        ++cohort_mislabeled_as_cycle.dependent_payoff_rejected_cycles;
        EXPECT_FALSE(cohort_mislabeled_as_cycle.valid());

        auto impossible_cohort_classification = with_rejected_cohort;
        impossible_cohort_classification.dependent_cohort_candidates = 0u;
        EXPECT_FALSE(impossible_cohort_classification.valid());
    }

    /**
     * @brief A watchdog renews on authority progress, not activity churn.
     *
     * A full device histogram can be drained only after the preceding
     * movement publishes. Rotating that RCU bank resets its row count but is
     * still monotonic progress. Conversely, activity enum changes do not prove
     * that a stuck proposal or transfer advanced and must not renew a timeout.
     */
    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        ProgressStampOrdersQueuedHistogramAndMovementEdges)
    {
        MoEOptimizationStatus initial{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity =
                MoEOptimizationActivityState::ReconcilingDemand,
            .demand_window = {
                .generation = 1u,
                .collected_routed_rows = 2u,
                .capacity_routed_rows = 4u,
            },
            .published_progress_generation = 3u,
            .reconciled_progress_generation = 2u,
        };
        const MoEOptimizationProgressStamp initial_stamp =
            initial.progressStamp();
        EXPECT_EQ(
            initial_stamp.relationTo(initial_stamp),
            MoEOptimizationProgressRelation::Unchanged);

        auto rows_advanced = initial;
        rows_advanced.demand_window.collected_routed_rows = 4u;
        EXPECT_EQ(
            rows_advanced.progressStamp().relationTo(initial_stamp),
            MoEOptimizationProgressRelation::Advanced);

        auto bank_rotated = rows_advanced;
        bank_rotated.demand_window.generation = 2u;
        bank_rotated.demand_window.collected_routed_rows = 0u;
        EXPECT_EQ(
            bank_rotated.progressStamp().relationTo(
                rows_advanced.progressStamp()),
            MoEOptimizationProgressRelation::Advanced)
            << "A new RCU bank is progress even though its occupancy resets";

        auto wave_completed = bank_rotated;
        wave_completed.published_movement_waves = 1u;
        wave_completed.completed_movement = {
            .transactions = 1u,
            .commands = 2u,
            .physical_bytes = 4096u,
            .promotions = 1u,
            .demotions = 1u,
        };
        EXPECT_EQ(
            wave_completed.progressStamp().relationTo(
                bank_rotated.progressStamp()),
            MoEOptimizationProgressRelation::Advanced);

        auto activity_only = wave_completed;
        activity_only.activity =
            MoEOptimizationActivityState::ExchangingProposal;
        EXPECT_EQ(
            activity_only.progressStamp().relationTo(
                wave_completed.progressStamp()),
            MoEOptimizationProgressRelation::Unchanged)
            << "Activity churn cannot conceal a stalled lifecycle edge";

        auto transaction_regressed = wave_completed;
        transaction_regressed.completed_movement.transactions = 0u;
        EXPECT_EQ(
            transaction_regressed.progressStamp().relationTo(
                wave_completed.progressStamp()),
            MoEOptimizationProgressRelation::Regressed);

        auto demand_regressed = rows_advanced;
        demand_regressed.demand_window.collected_routed_rows = 1u;
        EXPECT_EQ(
            demand_regressed.progressStamp().relationTo(
                rows_advanced.progressStamp()),
            MoEOptimizationProgressRelation::Regressed);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        PassiveProposalReceiveDoesNotOwnPublicationDeadline)
    {
        EXPECT_FALSE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Idle));
        EXPECT_FALSE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Receiving))
            << "A preposted peer mailbox may span calibration and idle serving";
        EXPECT_FALSE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::AwaitingValidation))
            << "The coordinator's active publication deadline covers local semantic adoption";
        EXPECT_TRUE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Publishing))
            << "An initiated coordinator send must retain the canonical timeout";
        EXPECT_TRUE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Acknowledging))
            << "A decoded peer generation must finish its readiness acknowledgement";
        EXPECT_FALSE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Failed));
        EXPECT_FALSE(moeOverlayProposalOwnsProgressDeadline(
            MoEOverlayMPIResidencyProposalPublisherState::Stopped));
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        TicketsRemainLiveWhilePrepareAndPublicationEventsArePending)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        auto old_ticket = authority->tryAcquireTicketSnapshot();
        ASSERT_TRUE(old_ticket.has_value());
        ASSERT_EQ((*old_ticket)->epoch, 1u);

        auto service = startService(authority, transport);
        const auto initial_optimization = service->optimizationStatus();
        EXPECT_EQ(
            initial_optimization.authority,
            MoEOptimizationAuthority::Host);
        EXPECT_EQ(
            initial_optimization.state,
            MoEOptimizationLifecycleState::Active);
        EXPECT_EQ(initial_optimization.published_movement_waves, 0u);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().stage_polls > 0; }));

        /* Admission and execution continue against epoch 1 during transfer. */
        for (int iteration = 0; iteration < 128; ++iteration)
        {
            auto ticket = authority->tryAcquireTicketSnapshot();
            ASSERT_TRUE(ticket.has_value());
            EXPECT_EQ((*ticket)->epoch, 1u);
        }
        EXPECT_EQ(authority->snapshot()->epoch, 1u);
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Staging);
        const auto staging_optimization = service->optimizationStatus();
        EXPECT_EQ(
            staging_optimization.activity,
            MoEOptimizationActivityState::MovingWeights);
        EXPECT_FALSE(staging_optimization.quiescentBetweenWaves());

        transport->setStageReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().prepare_begins == 1; }));
        EXPECT_EQ(authority->snapshot()->epoch, 1u);

        /* Preparation readiness is an independent event; it cannot be guessed. */
        for (int iteration = 0; iteration < 128; ++iteration)
        {
            auto ticket = authority->tryAcquireTicketSnapshot();
            ASSERT_TRUE(ticket.has_value());
            EXPECT_EQ((*ticket)->epoch, 1u);
        }

        transport->setPrepareReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().publication_begins == 1; }));
        EXPECT_EQ(authority->snapshot()->epoch, 1u)
            << "public admission must not advance during selector fan-out";

        transport->setPublicationReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));

        auto new_ticket = authority->tryAcquireTicketSnapshot();
        ASSERT_TRUE(new_ticket.has_value());
        EXPECT_EQ((*new_ticket)->epoch, 2u);
        EXPECT_EQ((*old_ticket)->epoch, 1u)
            << "The live old ticket must retain its exact engine bank";
        EXPECT_EQ(authority->pendingRetirementCount(), 1u);

        old_ticket.reset();
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().retirements == 1; }));
        EXPECT_EQ(authority->pendingRetirementCount(), 0u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();
        EXPECT_EQ(service->stats().committed_waves, 1u);
        EXPECT_EQ(authority->stats().committed_migrations, 4u);
        const auto moved_optimization = service->optimizationStatus();
        EXPECT_EQ(
            moved_optimization.state,
            MoEOptimizationLifecycleState::Active);
        EXPECT_EQ(moved_optimization.published_movement_waves, 1u);
        const auto authoritative_totals = authority->stats();
        EXPECT_EQ(moved_optimization.completed_movement.transactions, 1u);
        EXPECT_EQ(
            moved_optimization.completed_movement.commands,
            authoritative_totals.committed_migrations);
        EXPECT_EQ(
            moved_optimization.completed_movement.promotions,
            authoritative_totals.promotions);
        EXPECT_EQ(
            moved_optimization.completed_movement.demotions,
            authoritative_totals.demotions);
        EXPECT_EQ(
            moved_optimization.completed_movement.same_priority_moves,
            authoritative_totals.same_priority_moves);
        EXPECT_EQ(moved_optimization.completed_movement.physical_bytes, 0u)
            << "The structural transport deliberately owns no byte payload";
        const auto movement_ledger = authority->movementLedger();
        ASSERT_TRUE(movement_ledger.complete());
        EXPECT_EQ(
            movement_ledger.edges.size(),
            authoritative_totals.committed_migrations);
        ASSERT_EQ(movement_ledger.economy.size(), 1u)
            << "The host policy authority must retain one admitting proof per "
               "committed Dynamic wave";
        const auto &economy = movement_ledger.economy.front();
        EXPECT_TRUE(economy.valid());
        EXPECT_EQ(economy.authority, MoEOptimizationAuthority::Host);
        EXPECT_EQ(economy.transaction, 2u);
        EXPECT_EQ(economy.candidate_epoch, 2u);
        EXPECT_EQ(economy.command_count, movement_ledger.edges.size());
        EXPECT_GT(economy.cycle_count, 0u);

        ASSERT_TRUE(waitUntil(
            [&]
            { return service->optimizationStatus().quiescentBetweenWaves(); }))
            << "The worker never completed its post-publication demand poll";
        EXPECT_EQ(
            service->optimizationStatus().activity,
            MoEOptimizationActivityState::CollectingDemand);

        new_ticket.reset();
        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Stopped);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DeferredCapacityRetriesTheSameFrozenWindowWhileNewEvidenceAccumulates)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStartEnabled(false);
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().begin_calls >= 2; }));
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Deferred);

        /* These routes belong to the clean RCU bank installed by proposal. */
        const std::vector<uint64_t> later_counts{0, 0, 0, 0, 7, 0};
        recordObservedPrefillBatch(*histogram, 0, later_counts);
        EXPECT_EQ(histogram->activationCount(0, 4), 7u);

        transport->setStartEnabled(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));

        const auto observations = transport->observations();
        EXPECT_GE(observations.begin_calls, 3);
        EXPECT_EQ(observations.waves_started, 1);
        EXPECT_TRUE(observations.retained_identity)
            << "Backpressure must not replan or swap histogram generations";
        EXPECT_NE(observations.first_window, nullptr);
        EXPECT_EQ(authority->stats().checks, 1u)
            << "The frozen proposal is retried verbatim";
        EXPECT_EQ(histogram->activationCount(0, 4), 7u)
            << "Newer inference evidence must survive publication";
        EXPECT_GE(service->stats().deferred_attempts, 2u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        QueuedDemandCannotMasqueradeAsASettledBetweenWaveBoundary)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().publication_begins == 1; }));

        /* Queue a deliberately reversed hotness window while epoch two is
         * waiting on its publication event. The next poll must consume this
         * complete demand rather than briefly advertise an idle boundary. */
        const std::vector<uint64_t> reversed_counts{4, 200, 6, 180, 2, 8};
        recordObservedPrefillBatch(*histogram, 0, reversed_counts);
        ASSERT_TRUE(histogram->windowFull());

        /* Epoch two no longer needs staging. Hold epoch three at that event so
         * the typed public state can be observed without a timing race. */
        transport->setStageReady(false);
        transport->setPublicationReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            {
                return authority->snapshot()->epoch == 2u &&
                       transport->observations().begin_calls >= 2 &&
                       service->state() ==
                           MoEOverlayMaintenanceState::Staging;
            }));

        const auto queued = service->optimizationStatus();
        EXPECT_EQ(
            queued.activity,
            MoEOptimizationActivityState::MovingWeights);
        EXPECT_FALSE(queued.quiescentBetweenWaves())
            << "A queued post-publication wave was exposed as a stable timing "
               "cohort boundary";

        transport->setStageReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 3u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 2u; }));
        EXPECT_TRUE(service->healthy()) << service->failureMessage();
        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        GloballyDeferredStartedWaveAbortsAndRetriesTheSameFrozenTransaction)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setDeferDuringStage(true);
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            {
                const auto observed = transport->observations();
                return observed.begin_calls >= 2 && observed.aborts >= 1;
            }));
        EXPECT_EQ(authority->snapshot()->epoch, 1u);
        EXPECT_TRUE(transport->observations().retained_identity);
        EXPECT_EQ(authority->stats().checks, 1u)
            << "A globally deferred wave must retain rather than replan";
        EXPECT_GE(authority->stats().deferred_waves, 1u);

        transport->setDeferDuringStage(false);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 1u; }));

        const auto observed = transport->observations();
        EXPECT_GE(observed.waves_started, 2);
        EXPECT_GE(observed.aborts, 1);
        EXPECT_TRUE(observed.retained_identity);
        EXPECT_GE(service->stats().deferred_attempts, 1u);
        EXPECT_EQ(service->stats().committed_waves, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DistributedCoordinatorPublishesFrozenEvidenceBeforeStartingMovement)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto publisher =
            std::make_shared<ImmediateProposalPublisher>(true);
        auto service = startService(authority, transport, publisher);

        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 1u; }));
        ASSERT_NE(publisher->publishedProposal(), nullptr);
        EXPECT_EQ(publisher->publicationBegins(), 1);
        EXPECT_EQ(publisher->publicationCompletions(), 1);
        EXPECT_EQ(
            publisher->publishedProposal()
                ->plan.histogram_window->expert_counts,
            (std::vector<std::uint64_t>{90, 20, 80, 10, 100, 70}));
        EXPECT_EQ(service->stats().proposals_published, 1u);
        EXPECT_EQ(service->stats().proposals_received, 0u);
        EXPECT_EQ(authority->stats().checks, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        publisher->stopAndDrain();
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        ShutdownCompletesIrrevocablyPublishedDistributedGeneration)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto publisher =
            std::make_shared<ImmediateProposalPublisher>(true);
        publisher->setPublicationReady(false);
        auto service = startService(authority, transport, publisher);

        ASSERT_TRUE(waitUntil(
            [&]
            {
                return publisher->publicationBegins() == 1 &&
                       publisher->polls() > 0;
            }));

        /*
         * Once the coordinator has started publishing a frozen generation, a
         * peer may already have derived and staged it. Shutdown must therefore
         * finish the same generation locally before the root releases worker
         * ranks; discarding it would strand the peer in residency consensus.
         */
        std::atomic<bool> drain_returned{false};
        std::jthread drain_thread(
            [&]
            {
                service->stopAndDrain(
                    MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
                drain_returned.store(true, std::memory_order_release);
            });
        ASSERT_TRUE(waitUntil(
            [&]
            {
                return service->state() ==
                       MoEOverlayMaintenanceState::Draining;
            }));
        EXPECT_FALSE(drain_returned.load(std::memory_order_acquire));

        publisher->setPublicationReady(true);
        service->notifyMaintenanceProgress();
        drain_thread.join();

        EXPECT_TRUE(drain_returned.load(std::memory_order_acquire));
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Stopped);
        EXPECT_EQ(publisher->publicationCompletions(), 1);
        EXPECT_EQ(authority->snapshot()->epoch, 2u);
        EXPECT_EQ(service->stats().committed_waves, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();
        publisher->stopAndDrain();
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DistributedPeerIgnoresPartialLocalHistogramAndRearmsImmediately)
    {
        auto histogram = fullMovementHistogram();
        auto coordinator_histogram = fullMovementHistogram();
        auto coordinator_authority =
            dynamicAuthority(coordinator_histogram.get());
        const auto published_window =
            coordinator_authority->freezeAndRotateHistogramWindow();
        const auto root_transaction =
            coordinator_authority->proposeFromFrozenHistogramWindow(
                published_window);
        auto published_proposal = std::make_shared<
            const MoEOverlayDistributedResidencyProposal>(
            makeMoEOverlayDistributedResidencyProposal(
                coordinator_authority->exportAuthoritativeResidencyPlan(
                    root_transaction),
                root_transaction));
        ASSERT_TRUE(published_proposal->valid());

        /*
         * Adoption is an executable-identity operation, not a second policy
         * decision.  Prove the proposal alone reconstructs the root's exact
         * transaction before exercising the asynchronous service lifecycle;
         * otherwise a lifecycle timeout would hide which canonical field the
         * wire plan failed to carry.
         */
        auto reconstruction_histogram = fullMovementHistogram();
        auto reconstruction_authority =
            dynamicAuthority(reconstruction_histogram.get());
        const auto reconstructed_transaction =
            reconstruction_authority->adoptAuthoritativeResidencyPlan(
                published_proposal->plan);
        EXPECT_EQ(
            root_transaction.previous->layered_ownership,
            reconstructed_transaction.previous->layered_ownership);
        EXPECT_EQ(
            root_transaction.candidate->layered_ownership,
            reconstructed_transaction.candidate->layered_ownership);
        ASSERT_EQ(
            root_transaction.candidate->placement_plan->routed_tiers.size(),
            reconstructed_transaction.candidate->placement_plan->routed_tiers
                .size());
        for (size_t tier_idx = 0;
             tier_idx <
             root_transaction.candidate->placement_plan->routed_tiers.size();
             ++tier_idx)
        {
            const auto &root_tier =
                root_transaction.candidate->placement_plan->routed_tiers[
                    tier_idx];
            const auto &reconstructed_tier = reconstructed_transaction
                                                 .candidate->placement_plan
                                                 ->routed_tiers[tier_idx];
            EXPECT_EQ(root_tier.name, reconstructed_tier.name);
            EXPECT_EQ(root_tier.domain, reconstructed_tier.domain);
            EXPECT_EQ(root_tier.priority, reconstructed_tier.priority);
            EXPECT_EQ(
                root_tier.max_experts_per_layer,
                reconstructed_tier.max_experts_per_layer);
            EXPECT_EQ(
                root_tier.memory_budget_bytes,
                reconstructed_tier.memory_budget_bytes);
            EXPECT_EQ(root_tier.fallback, reconstructed_tier.fallback);
            EXPECT_EQ(
                root_tier.resolved_live_experts_per_layer,
                reconstructed_tier.resolved_live_experts_per_layer);
        }
        ASSERT_EQ(
            root_transaction.candidate->owner_map.owners().size(),
            reconstructed_transaction.candidate->owner_map.owners().size());
        for (size_t owner_idx = 0;
             owner_idx < root_transaction.candidate->owner_map.owners().size();
             ++owner_idx)
        {
            const auto &root_owner =
                root_transaction.candidate->owner_map.owners()[owner_idx];
            const auto &reconstructed_owner =
                reconstructed_transaction.candidate->owner_map.owners()[
                    owner_idx];
            EXPECT_EQ(root_owner.layer_idx, reconstructed_owner.layer_idx);
            EXPECT_EQ(root_owner.expert_id, reconstructed_owner.expert_id);
            EXPECT_EQ(root_owner.tier_idx, reconstructed_owner.tier_idx);
            EXPECT_EQ(
                root_owner.owner_participant,
                reconstructed_owner.owner_participant);
            EXPECT_EQ(root_owner.device, reconstructed_owner.device);
            EXPECT_EQ(root_owner.resident, reconstructed_owner.resident);
            EXPECT_EQ(root_owner.tier_name, reconstructed_owner.tier_name);
            EXPECT_EQ(root_owner.domain_name, reconstructed_owner.domain_name);
            EXPECT_EQ(
                root_owner.domain_participant_index,
                reconstructed_owner.domain_participant_index);
            EXPECT_EQ(
                root_owner.owner_world_rank,
                reconstructed_owner.owner_world_rank);
            EXPECT_EQ(
                root_owner.owner_world_rank_known,
                reconstructed_owner.owner_world_rank_known);
            EXPECT_EQ(root_owner.address, reconstructed_owner.address);
        }
        EXPECT_EQ(
            fingerprintMoEOverlayResidencyExecutionPlan(root_transaction),
            fingerprintMoEOverlayResidencyExecutionPlan(
                reconstructed_transaction));

        (void)histogram->freezeAndRotateWindow();
        ASSERT_FALSE(histogram->windowFull());

        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setPrepareReady(true);
        transport->setPublicationReady(true);
        auto publisher = std::make_shared<ImmediateProposalPublisher>(
            false,
            published_proposal);
        auto service = startService(authority, transport, publisher);

        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 1u; }));
        EXPECT_EQ(publisher->deliveries(), 1);
        EXPECT_EQ(publisher->receiveRearms(), 1);
        EXPECT_EQ(service->stats().proposals_received, 1u);
        EXPECT_EQ(service->stats().proposal_receives_rearmed, 1u);
        EXPECT_FALSE(histogram->windowFull())
            << "A peer must not rotate or substitute its partial local evidence";
        EXPECT_EQ(
            histogram->windowGeneration(),
            1u);
        EXPECT_EQ(authority->stats().checks, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
        publisher->stopAndDrain();
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        StaticPolicyChecksImmobilityWithoutCallingPhysicalTransport)
    {
        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = threeTierPlan(
                    RoutedExpertResidencyPolicy::StaticById),
                .model_metadata = modelMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "device-free-test",
            });
        auto transport = std::make_shared<ControlledTransport>();
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().static_no_movement == 1u; }));
        EXPECT_EQ(transport->observations().begin_calls, 0);
        EXPECT_EQ(authority->stats().checks, 1u);
        EXPECT_EQ(authority->stats().static_no_movement_checks, 1u);
        EXPECT_EQ(authority->stats().committed_migrations, 0u);
        EXPECT_EQ(service->stats().proposals, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();
        const auto optimization = service->optimizationStatus();
        EXPECT_EQ(
            optimization.state,
            MoEOptimizationLifecycleState::MovementDisabled);
        EXPECT_EQ(optimization.published_movement_waves, 0u);
        EXPECT_EQ(optimization.completed_movement.transactions, 0u);
        EXPECT_EQ(optimization.completed_movement.commands, 0u);

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        PhysicalTransportFailureIsFatalAndNeverPublishesCandidate)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setFailStart(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil([&] { return !service->healthy(); }));
        const auto failure_poll = service->stats().poll_iterations;
        ASSERT_TRUE(waitUntil(
            [&]
            {
                return service->stats().poll_iterations >=
                       failure_poll + 8u;
            }));
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Failed);
        EXPECT_EQ(
            service->failureMessage(),
            "injected physical transport failure");
        EXPECT_EQ(authority->snapshot()->epoch, 1u);
        EXPECT_EQ(authority->stats().stage_failures, 1u);
        EXPECT_EQ(service->stats().fatal_failures, 1u);
        EXPECT_EQ(transport->observations().begin_calls, 1)
            << "A fatal transport result must clear retry intent before the "
               "worker publishes its Failed state";
        EXPECT_EQ(transport->observations().waves_started, 0);
        const auto optimization = service->optimizationStatus();
        EXPECT_EQ(
            optimization.state,
            MoEOptimizationLifecycleState::Failed);
        EXPECT_FALSE(optimization.diagnostic.empty());

        service->stopAndDrain(
            MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
    }

} // namespace llaminar2::test
