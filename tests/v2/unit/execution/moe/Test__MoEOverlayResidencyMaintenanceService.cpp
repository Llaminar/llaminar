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
#include "execution/moe/MoEOverlayMPIHistogramPublisher.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
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
            config.top_k = 2;
            config.window_size = 4;
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
            auto histogram =
                std::make_shared<DecodeExpertHistogram>(std::move(config));
            const std::vector<uint64_t> counts{90, 20, 80, 10, 100, 70};
            histogram->mergeLayerCounts(0, counts.data(), 6, false);
            histogram->recordTokenBoundary(0, 4);
            return histogram;
        }

        /** @brief Complete measured-policy fixture for protocol-only movement tests. */
        std::shared_ptr<const MoERoutedTierServiceProfile> serviceProfile()
        {
            auto profile = std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "maintenance-service-profile-v1";
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
         * continue while neither staging nor commit is permitted to advance.
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

                /** @brief Record commit enqueue without making it ready. */
                bool beginCommit(std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->commit_begins_;
                    owner_->cv_.notify_all();
                    return true;
                }

                /** @brief Poll the independently controlled commit event. */
                MoEOverlayResidencyWaveProgress pollCommit(
                    std::string *) noexcept override
                {
                    std::lock_guard<std::mutex> lock(owner_->mutex_);
                    ++owner_->commit_polls_;
                    owner_->cv_.notify_all();
                    return owner_->commit_ready_
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
                int commit_begins = 0;
                int commit_polls = 0;
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

            /** @brief Enable the exact inactive-bank commit event. */
            void setCommitReady(bool ready)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                commit_ready_ = ready;
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
                    .commit_begins = commit_begins_,
                    .commit_polls = commit_polls_,
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
            bool commit_ready_ = false;
            bool fail_start_ = false;
            int begin_calls_ = 0;
            int waves_started_ = 0;
            int stage_polls_ = 0;
            int commit_begins_ = 0;
            int commit_polls_ = 0;
            int aborts_ = 0;
            int retirements_ = 0;
            uint64_t first_generation_ = 0;
            const DecodeExpertHistogramWindow *first_window_ = nullptr;
            bool retained_identity_ = true;
        };

        /** @brief Immediate thread-safe frozen-window publisher for CPU tests. */
        class ImmediateHistogramPublisher final
            : public IMoEOverlayHistogramPublisher
        {
        public:
            /**
             * @brief Construct a coordinator or an initially armed peer.
             * @param coordinator Whether this process owns routing evidence.
             * @param peer_window First authenticated peer window, when any.
             */
            ImmediateHistogramPublisher(
                bool coordinator,
                std::shared_ptr<const DecodeExpertHistogramWindow> peer_window = {})
                : coordinator_(coordinator),
                  peer_window_(std::move(peer_window)),
                  active_(!coordinator)
            {
            }

            /** @brief Capture the exact coordinator window and begin its send. */
            bool beginPublish(
                const DecodeExpertHistogramWindow &window,
                std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!coordinator_ || active_ || !window.valid())
                {
                    if (error)
                        *error = "invalid immediate histogram publication";
                    return false;
                }
                published_window_ =
                    std::make_shared<DecodeExpertHistogramWindow>(window);
                active_ = true;
                ++publication_begins_;
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Complete a send or deliver one copied peer window. */
            MoEOverlayResidencyWaveProgress poll(
                std::shared_ptr<const DecodeExpertHistogramWindow> *received,
                std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (received)
                    received->reset();
                if (!active_)
                {
                    if (error)
                        *error = "immediate histogram lane is idle";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (coordinator_)
                {
                    active_ = false;
                    ++publication_completions_;
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (!peer_window_)
                    return MoEOverlayResidencyWaveProgress::Pending;

                if (received)
                    *received = std::move(peer_window_);
                else
                    peer_window_.reset();
                active_ = false;
                ++deliveries_;
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Arm the peer for another generation immediately. */
            bool armReceive(std::string *error) override
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (coordinator_ || active_)
                {
                    if (error)
                        *error = "invalid immediate histogram rearm";
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

            /** @return Number of authenticated peer window deliveries. */
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

            /** @return Coordinator window copied at beginPublish(). */
            std::shared_ptr<const DecodeExpertHistogramWindow>
            publishedWindow() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return published_window_;
            }

        private:
            bool coordinator_ = false;
            mutable std::mutex mutex_;
            std::shared_ptr<const DecodeExpertHistogramWindow> peer_window_;
            std::shared_ptr<const DecodeExpertHistogramWindow>
                published_window_;
            bool active_ = false;
            int publication_begins_ = 0;
            int publication_completions_ = 0;
            int deliveries_ = 0;
            int receive_rearms_ = 0;
        };

        /** @brief Start a service with a fast, non-semantic test cadence. */
        std::unique_ptr<MoEOverlayResidencyMaintenanceService> startService(
            std::shared_ptr<MoEOverlayResidencyAuthority> authority,
            std::shared_ptr<ControlledTransport> transport,
            std::shared_ptr<IMoEOverlayHistogramPublisher> publisher = {})
        {
            return std::make_unique<
                MoEOverlayResidencyMaintenanceService>(
                MoEOverlayResidencyMaintenanceService::Config{
                    .authority = std::move(authority),
                    .transport = std::move(transport),
                    .histogram_publisher = std::move(publisher),
                    .idle_poll_interval = 100us,
                    .perf_device = "device-free-test",
                });
        }
    } // namespace

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        PassiveHistogramReceiveDoesNotOwnPublicationDeadline)
    {
        EXPECT_FALSE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Idle));
        EXPECT_FALSE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Receiving))
            << "A preposted peer mailbox may span calibration and idle serving";
        EXPECT_TRUE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Publishing))
            << "An initiated coordinator send must retain the canonical timeout";
        EXPECT_TRUE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Acknowledging))
            << "A decoded peer generation must finish its readiness acknowledgement";
        EXPECT_FALSE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Failed));
        EXPECT_FALSE(moeOverlayHistogramOwnsProgressDeadline(
            MoEOverlayMPIHistogramPublisherState::Stopped));
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        TicketsRemainLiveWhileStageAndCommitEventsArePending)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        auto old_ticket = authority->tryAcquireTicketSnapshot();
        ASSERT_TRUE(old_ticket.has_value());
        ASSERT_EQ((*old_ticket)->epoch, 1u);

        auto service = startService(authority, transport);
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

        transport->setStageReady(true);
        service->notifyMaintenanceProgress();
        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().commit_begins == 1; }));
        EXPECT_EQ(authority->snapshot()->epoch, 1u);

        /* Commit readiness is a second independent event; it cannot be guessed. */
        for (int iteration = 0; iteration < 128; ++iteration)
        {
            auto ticket = authority->tryAcquireTicketSnapshot();
            ASSERT_TRUE(ticket.has_value());
            EXPECT_EQ((*ticket)->epoch, 1u);
        }

        transport->setCommitReady(true);
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

        new_ticket.reset();
        service->stopAndDrain();
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
        transport->setCommitReady(true);
        auto service = startService(authority, transport);

        ASSERT_TRUE(waitUntil(
            [&]
            { return transport->observations().begin_calls >= 2; }));
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Deferred);

        /* These routes belong to the clean RCU bank installed by proposal. */
        const std::vector<uint64_t> later_counts{0, 0, 0, 0, 7, 0};
        histogram->mergeLayerCounts(0, later_counts.data(), 6, false);
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

        service->stopAndDrain();
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
        transport->setCommitReady(true);
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

        service->stopAndDrain();
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DistributedCoordinatorPublishesFrozenEvidenceBeforeStartingMovement)
    {
        auto histogram = fullMovementHistogram();
        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setCommitReady(true);
        auto publisher =
            std::make_shared<ImmediateHistogramPublisher>(true);
        auto service = startService(authority, transport, publisher);

        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 1u; }));
        ASSERT_NE(publisher->publishedWindow(), nullptr);
        EXPECT_EQ(publisher->publicationBegins(), 1);
        EXPECT_EQ(publisher->publicationCompletions(), 1);
        EXPECT_EQ(
            publisher->publishedWindow()->expert_counts,
            (std::vector<std::uint64_t>{90, 20, 80, 10, 100, 70}));
        EXPECT_EQ(service->stats().histogram_windows_published, 1u);
        EXPECT_EQ(service->stats().histogram_windows_received, 0u);
        EXPECT_EQ(authority->stats().checks, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain();
        publisher->stopAndDrain();
    }

    TEST(
        Test__MoEOverlayResidencyMaintenanceService,
        DistributedPeerIgnoresPartialLocalHistogramAndRearmsImmediately)
    {
        auto histogram = fullMovementHistogram();
        auto published_window =
            std::make_shared<const DecodeExpertHistogramWindow>(
                histogram->freezeAndRotateWindow());
        ASSERT_TRUE(published_window->valid());
        ASSERT_FALSE(histogram->windowFull());

        auto authority = dynamicAuthority(histogram.get());
        auto transport = std::make_shared<ControlledTransport>();
        transport->setStageReady(true);
        transport->setCommitReady(true);
        auto publisher = std::make_shared<ImmediateHistogramPublisher>(
            false,
            published_window);
        auto service = startService(authority, transport, publisher);

        ASSERT_TRUE(waitUntil(
            [&]
            { return authority->snapshot()->epoch == 2u; }));
        ASSERT_TRUE(waitUntil(
            [&]
            { return service->stats().committed_waves == 1u; }));
        EXPECT_EQ(publisher->deliveries(), 1);
        EXPECT_EQ(publisher->receiveRearms(), 1);
        EXPECT_EQ(service->stats().histogram_windows_received, 1u);
        EXPECT_EQ(service->stats().histogram_receives_rearmed, 1u);
        EXPECT_FALSE(histogram->windowFull())
            << "A peer must not rotate or substitute its partial local evidence";
        EXPECT_EQ(
            histogram->windowGeneration(),
            published_window->generation + 1);
        EXPECT_EQ(authority->stats().checks, 1u);
        EXPECT_TRUE(service->healthy()) << service->failureMessage();

        service->stopAndDrain();
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

        service->stopAndDrain();
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
        EXPECT_EQ(service->state(), MoEOverlayMaintenanceState::Failed);
        EXPECT_EQ(
            service->failureMessage(),
            "injected physical transport failure");
        EXPECT_EQ(authority->snapshot()->epoch, 1u);
        EXPECT_EQ(authority->stats().stage_failures, 1u);
        EXPECT_EQ(service->stats().fatal_failures, 1u);
        EXPECT_EQ(transport->observations().waves_started, 0);

        service->stopAndDrain();
    }

} // namespace llaminar2::test
