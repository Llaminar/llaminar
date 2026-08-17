/**
 * @file Test__MoEOverlayDistributedResidencyTransport.cpp
 * @brief CPU-only adversarial tests for globally coordinated residency waves.
 *
 * An in-process asynchronous lane models independently progressing MPI ranks.
 * These tests prove that local backpressure, physical failure, and commit-start
 * failure are exchanged by every participant, that no peer is stranded in a
 * different phase, that a deferred transaction can be retried unchanged, and
 * that no local old bank retires before every rank reaches its lease fence.
 */

#include "execution/moe/MoEOverlayDistributedResidencyTransport.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @brief Build one process-owned whole-expert execution domain. */
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

        /** @brief Build one thermally ordered routed-expert tier. */
        RoutedExpertTier tier(
            std::string name,
            std::string domain_name,
            int priority,
            int capacity,
            bool cold_remainder = false)
        {
            RoutedExpertTier result;
            result.name = std::move(name);
            result.domain = std::move(domain_name);
            result.priority = priority;
            result.max_experts_per_layer = capacity;
            result.fallback = cold_remainder;
            return result;
        }

        /** @brief Resolve a three-rank topology independent of rank/device order. */
        MoERoutedExpertPlacementPlan threeRankPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Random;
            plan.domains = {
                domain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(1, 1, "node-a"),
                    2,
                    CollectiveBackendType::NCCL),
                domain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(0, 0, "node-a"),
                    0,
                    CollectiveBackendType::RCCL),
                domain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(1, "node-a"),
                    1,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                tier("hot", "cuda_hot", 0, 2),
                tier("warm", "rocm_warm", 1, 2),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /** @brief Minimal model geometry that still produces a three-tier swap. */
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

        /** @brief Own all authority dependencies behind one stable transaction. */
        struct TransactionFixture
        {
            std::unique_ptr<DecodeExpertHistogram> histogram;
            std::unique_ptr<MoEOverlayResidencyAuthority> authority;
            MoEOverlayResidencyTransaction transaction;
        };

        /** @brief Build one non-empty, valid histogram-driven transaction. */
        TransactionFixture makeTransaction()
        {
            DecodeExpertHistogramConfig histogram_config;
            histogram_config.num_layers = 1;
            histogram_config.num_experts = 6;
            histogram_config.top_k = 2;
            histogram_config.window_size = 4;
            histogram_config.sockets = {
                DeviceId::cuda(1),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            histogram_config.ownership =
                MoELayeredExpertOwnership::uniform(
                    1,
                    3,
                    {0, 0, 1, 1, 2, 2});

            TransactionFixture fixture;
            fixture.histogram =
                std::make_unique<DecodeExpertHistogram>(histogram_config);
            const std::vector<std::uint64_t> counts{1, 2, 3, 4, 100, 90};
            fixture.histogram->mergeLayerCounts(
                0,
                counts.data(),
                static_cast<int>(counts.size()),
                false);
            fixture.authority =
                std::make_unique<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{
                        .initial_plan = threeRankPlan(),
                        .model_metadata = modelMetadata(),
                        .maintenance_mode =
                            MoERebalanceRuntimeMode::Dynamic,
                        .histogram = fixture.histogram.get(),
                        .perf_device = "distributed_transport_cpu_test",
                    });
            fixture.transaction = fixture.authority->proposeFromHistogram();
            if (!fixture.transaction.valid() || fixture.transaction.empty())
            {
                throw std::logic_error(
                    "Distributed transport fixture did not produce movement");
            }
            return fixture;
        }

        /**
         * @brief Shared generation-indexed vote exchange for independent lanes.
         *
         * A fast rank may complete generation N and submit N+1 before another
         * rank has locally consumed N, matching ordered non-blocking MPI
         * collectives. The bus retains each tiny generation for the test life.
         */
        class InProcessConsensusBus final
        {
        public:
            /** @brief Allocate one fixed-width participant set. */
            explicit InProcessConsensusBus(int world_size)
                : world_size_(world_size)
            {
                if (world_size_ < 2)
                    throw std::invalid_argument("Consensus bus requires peers");
            }

            /**
             * @brief Submit one rank vote into an exact collective generation.
             * @return False for invalid geometry or a duplicate submission.
             */
            bool submit(
                std::size_t generation,
                int world_rank,
                const MoEOverlayDistributedResidencyVote &vote)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (world_rank < 0 || world_rank >= world_size_)
                    return false;
                if (generations_.size() <= generation)
                    generations_.resize(generation + 1);
                auto &slot = generations_[generation];
                if (slot.votes.empty())
                {
                    slot.votes.resize(
                        static_cast<std::size_t>(world_size_));
                }
                auto &rank_vote =
                    slot.votes[static_cast<std::size_t>(world_rank)];
                if (rank_vote.has_value())
                    return false;
                rank_vote = vote;
                return true;
            }

            /**
             * @brief Poll one generation and copy rank-ordered votes when full.
             * @param generation Lane-local ordered collective sequence number.
             * @param votes Output records on Ready.
             * @return Pending until every rank has submitted, then Ready.
             */
            MoEOverlayResidencyWaveProgress poll(
                std::size_t generation,
                std::vector<MoEOverlayDistributedResidencyVote> *votes)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation >= generations_.size())
                    return MoEOverlayResidencyWaveProgress::Pending;
                const auto &slot = generations_[generation];
                if (slot.votes.size() !=
                        static_cast<std::size_t>(world_size_) ||
                    std::any_of(
                        slot.votes.begin(),
                        slot.votes.end(),
                        [](const auto &vote) { return !vote.has_value(); }))
                {
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                if (votes)
                {
                    votes->clear();
                    votes->reserve(slot.votes.size());
                    for (const auto &vote : slot.votes)
                        votes->push_back(*vote);
                }
                return MoEOverlayResidencyWaveProgress::Ready;
            }

        private:
            /** @brief Optional rank records for one ordered collective. */
            struct Generation
            {
                std::vector<std::optional<
                    MoEOverlayDistributedResidencyVote>> votes;
            };

            int world_size_ = 0;
            std::mutex mutex_;
            std::vector<Generation> generations_;
        };

        /** @brief Per-rank non-blocking view over the in-process consensus bus. */
        class InProcessConsensusLane final
            : public IMoEOverlayResidencyConsensusLane
        {
        public:
            /** @brief Bind one exact rank to the shared generation sequence. */
            InProcessConsensusLane(
                std::shared_ptr<InProcessConsensusBus> bus,
                int world_rank,
                int world_size)
                : bus_(std::move(bus)),
                  world_rank_(world_rank),
                  world_size_(world_size)
            {
            }

            /** @brief Submit one local vote without waiting for peers. */
            bool begin(
                const MoEOverlayDistributedResidencyVote &vote,
                std::string *error) override
            {
                if (active_ || !vote.valid(world_size_) ||
                    vote.world_rank != world_rank_ ||
                    !bus_->submit(next_generation_, world_rank_, vote))
                {
                    if (error)
                        *error = "In-process consensus rejected local vote";
                    return false;
                }
                active_generation_ = next_generation_++;
                active_ = true;
                if (error)
                    error->clear();
                return true;
            }

            /** @brief Query the shared generation once without waiting. */
            MoEOverlayResidencyWaveProgress poll(
                std::vector<MoEOverlayDistributedResidencyVote> *votes,
                std::string *error) override
            {
                if (!active_)
                {
                    if (error)
                        *error = "In-process consensus has no active exchange";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                const auto progress = bus_->poll(active_generation_, votes);
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    active_ = false;
                if (error)
                    error->clear();
                return progress;
            }

            /** @return Whether this rank has consumed its active generation. */
            [[nodiscard]] bool idle() const noexcept override
            {
                return !active_;
            }

            /** @return Rank fixed at lane construction. */
            [[nodiscard]] int worldRank() const noexcept override
            {
                return world_rank_;
            }

            /** @return Participant count fixed at lane construction. */
            [[nodiscard]] int worldSize() const noexcept override
            {
                return world_size_;
            }

        private:
            std::shared_ptr<InProcessConsensusBus> bus_;
            int world_rank_ = -1;
            int world_size_ = 0;
            std::size_t next_generation_ = 0;
            std::size_t active_generation_ = 0;
            bool active_ = false;
        };

        /** @brief Scriptable process-local physical transport and event wave. */
        class ScriptedLocalTransport final
            : public IMoEOverlayResidencyTransport
        {
        public:
            /** @brief One local event-polled wave retaining its owner fixture. */
            class Wave final : public IMoEOverlayResidencyWave
            {
            public:
                /** @brief Bind wave observations to one stable local transport. */
                explicit Wave(ScriptedLocalTransport *owner) : owner_(owner) {}

                /** @brief Return the currently scripted physical stage event. */
                MoEOverlayResidencyWaveProgress pollStage(
                    std::string *error) noexcept override
                {
                    ++owner_->stage_polls;
                    if (owner_->stage_progress ==
                            MoEOverlayResidencyWaveProgress::Failed &&
                        error)
                    {
                        *error = "injected local physical stage failure";
                    }
                    return owner_->stage_progress;
                }

                /** @brief Record and return the scripted local commit enqueue. */
                bool beginCommit(std::string *error) noexcept override
                {
                    ++owner_->commit_begins;
                    if (!owner_->commit_begin_ok && error)
                        *error = "injected local commit enqueue failure";
                    return owner_->commit_begin_ok;
                }

                /** @brief Return the currently scripted local bank event. */
                MoEOverlayResidencyWaveProgress pollCommit(
                    std::string *error) noexcept override
                {
                    ++owner_->commit_polls;
                    if (owner_->commit_progress !=
                            MoEOverlayResidencyWaveProgress::Ready &&
                        owner_->commit_progress !=
                            MoEOverlayResidencyWaveProgress::Pending &&
                        error)
                    {
                        *error = "injected local commit event failure";
                    }
                    return owner_->commit_progress;
                }

                /** @brief Record asynchronous cleanup initiation. */
                void abortStaged() noexcept override
                {
                    ++owner_->aborts;
                }

                /** @brief This CPU fixture's cleanup edge is immediately ready. */
                MoEOverlayResidencyWaveProgress pollAbort(
                    std::string *) noexcept override
                {
                    ++owner_->abort_polls;
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                /** @brief Record old-bank retirement after publication. */
                void retirePrevious() noexcept override
                {
                    ++owner_->retirements;
                }

            private:
                ScriptedLocalTransport *owner_ = nullptr;
            };

            /** @brief Return the scripted start and retain transaction identity. */
            MoEOverlayResidencyStageStart beginStage(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                ++begin_calls;
                transaction_fingerprints.push_back(
                    fingerprintMoEOverlayResidencyTransaction(transaction));
                switch (start_status)
                {
                case MoEOverlayResidencyStageStartStatus::Started:
                    return {
                        .status = start_status,
                        .wave = std::make_unique<Wave>(this),
                    };
                case MoEOverlayResidencyStageStartStatus::Deferred:
                    return {
                        .status = start_status,
                        .error = "injected transient shadow-slot pressure",
                    };
                case MoEOverlayResidencyStageStartStatus::Failed:
                    return {
                        .status = start_status,
                        .cleanup_wave =
                            failure_owns_cleanup
                                ? std::make_unique<Wave>(this)
                                : nullptr,
                        .error = "injected local stage-start failure",
                    };
                }
                return {
                    .status = MoEOverlayResidencyStageStartStatus::Failed,
                    .error = "invalid scripted start status",
                };
            }

            MoEOverlayResidencyStageStartStatus start_status =
                MoEOverlayResidencyStageStartStatus::Started;
            MoEOverlayResidencyWaveProgress stage_progress =
                MoEOverlayResidencyWaveProgress::Ready;
            bool commit_begin_ok = true;
            MoEOverlayResidencyWaveProgress commit_progress =
                MoEOverlayResidencyWaveProgress::Ready;
            bool failure_owns_cleanup = false;
            int begin_calls = 0;
            int stage_polls = 0;
            int commit_begins = 0;
            int commit_polls = 0;
            int aborts = 0;
            int abort_polls = 0;
            int retirements = 0;
            std::vector<MoEOverlayResidencyTransactionFingerprint>
                transaction_fingerprints;
        };

        /** @brief Dependencies and facades for one three-rank process simulation. */
        struct DistributedTransportFixture
        {
            static constexpr int kWorldSize = 3;
            std::shared_ptr<InProcessConsensusBus> bus =
                std::make_shared<InProcessConsensusBus>(kWorldSize);
            std::vector<std::unique_ptr<ScriptedLocalTransport>> locals;
            std::vector<std::shared_ptr<InProcessConsensusLane>> lanes;
            std::vector<std::unique_ptr<
                MoEOverlayDistributedResidencyTransport>> transports;

            /** @brief Materialize every rank with independent local state. */
            DistributedTransportFixture()
            {
                for (int rank = 0; rank < kWorldSize; ++rank)
                {
                    locals.push_back(
                        std::make_unique<ScriptedLocalTransport>());
                    lanes.push_back(
                        std::make_shared<InProcessConsensusLane>(
                            bus,
                            rank,
                            kWorldSize));
                    transports.push_back(std::make_unique<
                        MoEOverlayDistributedResidencyTransport>(
                        MoEOverlayDistributedResidencyTransport::Config{
                            .local_transport = locals.back().get(),
                            .consensus = lanes.back(),
                            .perf_device = "in_process_three_tier",
                        }));
                }
            }
        };

        /** @brief One terminal progress value and its rank-local diagnostic. */
        struct TerminalProgress
        {
            MoEOverlayResidencyWaveProgress progress =
                MoEOverlayResidencyWaveProgress::Pending;
            std::string error;
        };

        /** @brief Poll all independent stage waves until each is terminal. */
        std::vector<TerminalProgress> finishStages(
            std::vector<MoEOverlayResidencyStageStart> &starts)
        {
            std::vector<TerminalProgress> results(starts.size());
            for (int pass = 0; pass < 32; ++pass)
            {
                for (std::size_t rank = 0; rank < starts.size(); ++rank)
                {
                    if (results[rank].progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        results[rank].progress =
                            starts[rank].wave->pollStage(
                                &results[rank].error);
                    }
                }
                if (std::all_of(
                        results.begin(),
                        results.end(),
                        [](const auto &result)
                        {
                            return result.progress !=
                                   MoEOverlayResidencyWaveProgress::Pending;
                        }))
                {
                    return results;
                }
            }
            ADD_FAILURE() << "Distributed stage did not converge";
            return results;
        }

        /** @brief Poll all independent commit waves until each is terminal. */
        std::vector<TerminalProgress> finishCommits(
            std::vector<MoEOverlayResidencyStageStart> &starts)
        {
            std::vector<TerminalProgress> results(starts.size());
            for (int pass = 0; pass < 32; ++pass)
            {
                for (std::size_t rank = 0; rank < starts.size(); ++rank)
                {
                    if (results[rank].progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        results[rank].progress =
                            starts[rank].wave->pollCommit(
                                &results[rank].error);
                    }
                }
                if (std::all_of(
                        results.begin(),
                        results.end(),
                        [](const auto &result)
                        {
                            return result.progress !=
                                   MoEOverlayResidencyWaveProgress::Pending;
                        }))
                {
                    return results;
                }
            }
            ADD_FAILURE() << "Distributed commit did not converge";
            return results;
        }

        /** @brief Poll all published waves through the global lease fence. */
        std::vector<TerminalProgress> finishRetirementFences(
            std::vector<MoEOverlayResidencyStageStart> &starts)
        {
            std::vector<TerminalProgress> results(starts.size());
            for (int pass = 0; pass < 32; ++pass)
            {
                for (std::size_t rank = 0; rank < starts.size(); ++rank)
                {
                    if (results[rank].progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        results[rank].progress =
                            starts[rank].wave->pollRetirementFence(
                                &results[rank].error);
                    }
                }
                if (std::all_of(
                        results.begin(),
                        results.end(),
                        [](const auto &result)
                        {
                            return result.progress !=
                                   MoEOverlayResidencyWaveProgress::Pending;
                        }))
                {
                    return results;
                }
            }
            ADD_FAILURE() << "Distributed retirement fence did not converge";
            return results;
        }

        /** @brief Start the same exact transaction through every rank facade. */
        std::vector<MoEOverlayResidencyStageStart> startEveryRank(
            DistributedTransportFixture &distributed,
            const MoEOverlayResidencyTransaction &transaction)
        {
            std::vector<MoEOverlayResidencyStageStart> starts;
            for (auto &transport : distributed.transports)
                starts.push_back(transport->beginStage(transaction));
            return starts;
        }

        /** @brief Abort every unpublished wrapper and prove cleanup readiness. */
        void abortEveryRank(
            std::vector<MoEOverlayResidencyStageStart> &starts)
        {
            for (auto &start : starts)
                start.wave->abortStaged();
            for (auto &start : starts)
            {
                std::string error;
                EXPECT_EQ(
                    start.wave->pollAbort(&error),
                    MoEOverlayResidencyWaveProgress::Ready)
                    << error;
            }
        }
    } // namespace

    TEST(
        Test__MoEOverlayDistributedResidencyTransport,
        OneRankBackpressureDefersAllRanksAndRetriesIdenticalTransaction)
    {
        auto transaction = makeTransaction();
        DistributedTransportFixture distributed;
        distributed.locals[1]->start_status =
            MoEOverlayResidencyStageStartStatus::Deferred;

        auto first = startEveryRank(distributed, transaction.transaction);
        for (const auto &start : first)
        {
            ASSERT_EQ(
                start.status,
                MoEOverlayResidencyStageStartStatus::Started);
            ASSERT_NE(start.wave, nullptr);
        }
        const auto deferred = finishStages(first);
        for (const auto &result : deferred)
        {
            EXPECT_EQ(
                result.progress,
                MoEOverlayResidencyWaveProgress::Deferred)
                << result.error;
            EXPECT_TRUE(result.error.empty());
        }
        abortEveryRank(first);
        EXPECT_EQ(distributed.locals[0]->aborts, 1);
        EXPECT_EQ(distributed.locals[1]->aborts, 0);
        EXPECT_EQ(distributed.locals[2]->aborts, 1);

        distributed.locals[1]->start_status =
            MoEOverlayResidencyStageStartStatus::Started;
        auto retry = startEveryRank(distributed, transaction.transaction);
        const auto staged = finishStages(retry);
        for (const auto &result : staged)
            EXPECT_EQ(result.progress, MoEOverlayResidencyWaveProgress::Ready);
        for (auto &start : retry)
        {
            const auto interval = start.wave->completedStageInterval();
            ASSERT_TRUE(interval.has_value());
            EXPECT_TRUE(interval->valid());
            std::string error;
            EXPECT_TRUE(start.wave->beginCommit(&error)) << error;
        }
        const auto committed = finishCommits(retry);
        for (std::size_t rank = 0; rank < retry.size(); ++rank)
        {
            EXPECT_EQ(
                committed[rank].progress,
                MoEOverlayResidencyWaveProgress::Ready)
                << committed[rank].error;
            retry[rank].wave->markPublished();
        }
        const auto retirement_ready = finishRetirementFences(retry);
        for (std::size_t rank = 0; rank < retry.size(); ++rank)
        {
            EXPECT_EQ(
                retirement_ready[rank].progress,
                MoEOverlayResidencyWaveProgress::Ready)
                << retirement_ready[rank].error;
            retry[rank].wave->retirePrevious();
            EXPECT_TRUE(distributed.lanes[rank]->idle());
        }

        for (const auto &local : distributed.locals)
        {
            ASSERT_EQ(local->transaction_fingerprints.size(), 2u);
            EXPECT_EQ(
                local->transaction_fingerprints[0],
                local->transaction_fingerprints[1]);
            EXPECT_EQ(local->retirements, 1);
        }
        const auto rank_one_stats = distributed.transports[1]->stats();
        EXPECT_EQ(rank_one_stats.local_stage_deferred, 1u);
        EXPECT_EQ(rank_one_stats.local_stage_started, 1u);
        EXPECT_EQ(rank_one_stats.reservation_consensus_deferred, 1u);
        EXPECT_EQ(rank_one_stats.reservation_consensus_ready, 1u);
        EXPECT_EQ(rank_one_stats.stage_consensus_ready, 1u);
        EXPECT_EQ(rank_one_stats.commit_consensus_ready, 1u);
        EXPECT_EQ(rank_one_stats.retirement_consensus_started, 1u);
        EXPECT_EQ(rank_one_stats.retirement_consensus_ready, 1u);
        EXPECT_EQ(rank_one_stats.retirement_consensus_failed, 0u);
        EXPECT_EQ(rank_one_stats.waves_published, 1u);
        EXPECT_EQ(rank_one_stats.inference_thread_waits, 0u);
        EXPECT_EQ(rank_one_stats.blocking_synchronizations, 0u);
    }

    TEST(
        Test__MoEOverlayDistributedResidencyTransport,
        CommitEnqueueFailureIsGloballyVotedWithoutStrandingPeers)
    {
        auto transaction = makeTransaction();
        DistributedTransportFixture distributed;
        distributed.locals[1]->commit_begin_ok = false;

        auto starts = startEveryRank(distributed, transaction.transaction);
        const auto staged = finishStages(starts);
        for (const auto &result : staged)
            ASSERT_EQ(result.progress, MoEOverlayResidencyWaveProgress::Ready);

        for (auto &start : starts)
        {
            std::string error;
            EXPECT_TRUE(start.wave->beginCommit(&error)) << error;
        }
        const auto failed = finishCommits(starts);
        for (const auto &result : failed)
        {
            EXPECT_EQ(
                result.progress,
                MoEOverlayResidencyWaveProgress::Failed);
            EXPECT_NE(result.error.find("world rank 1"), std::string::npos)
                << result.error;
            EXPECT_NE(result.error.find("2103"), std::string::npos)
                << result.error;
        }
        abortEveryRank(starts);

        EXPECT_EQ(
            distributed.transports[1]->stats().local_commit_begin_failed,
            1u);
        for (std::size_t rank = 0; rank < starts.size(); ++rank)
        {
            EXPECT_EQ(
                distributed.transports[rank]
                    ->stats()
                    .commit_consensus_failed,
                1u);
            EXPECT_TRUE(distributed.lanes[rank]->idle());
        }
    }

    TEST(
        Test__MoEOverlayDistributedResidencyTransport,
        PartialLocalStartFailureCleanupIsOwnedUntilGlobalAbort)
    {
        auto transaction = makeTransaction();
        DistributedTransportFixture distributed;
        distributed.locals[2]->start_status =
            MoEOverlayResidencyStageStartStatus::Failed;
        distributed.locals[2]->failure_owns_cleanup = true;

        auto starts = startEveryRank(distributed, transaction.transaction);
        for (const auto &start : starts)
            ASSERT_EQ(start.status, MoEOverlayResidencyStageStartStatus::Started);
        const auto failed = finishStages(starts);
        for (const auto &result : failed)
        {
            EXPECT_EQ(
                result.progress,
                MoEOverlayResidencyWaveProgress::Failed);
            EXPECT_NE(result.error.find("world rank 2"), std::string::npos)
                << result.error;
            EXPECT_NE(result.error.find("2101"), std::string::npos)
                << result.error;
        }

        abortEveryRank(starts);
        EXPECT_EQ(distributed.locals[2]->aborts, 1)
            << "Partially enqueued cleanup ownership must not be dropped";
        EXPECT_EQ(distributed.locals[2]->abort_polls, 1);
        EXPECT_EQ(distributed.transports[2]->stats().local_stage_failed, 1u);
        for (const auto &lane : distributed.lanes)
            EXPECT_TRUE(lane->idle());
    }
} // namespace llaminar2::test
