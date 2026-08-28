/**
 * @file Test__MoEOverlayInferenceTransaction.cpp
 * @brief Adversarial CPU tests for ExpertOverlay transaction scheduling.
 *
 * These tests deliberately use no MPI runtime or accelerator. They certify the
 * protocol that will own registered MPI buffers and retained CUDA/ROCm graphs,
 * including fixed-slot backpressure, depth geometry, epoch monotonicity, and
 * exact terminal ordering before device integration is allowed to depend on it.
 */

#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <stdexcept>

namespace llaminar2::test
{
    namespace
    {
        MoEOverlayInferenceTopologyIdentity topology(int target_rank = 1)
        {
            return {
                .workspace_generation = 71,
                .topology_fingerprint_low = 0x1122334455667788ULL,
                .topology_fingerprint_high = 0x8877665544332211ULL,
                .source_world_rank = 0,
                .target_world_rank = target_rank,
            };
        }

        /** @brief Sealed CPU-only stand-in for coordinator lifecycle tests. */
        MoEOverlayInferenceCoordinatorGraphPlan coordinatorGraphPlan(
            int continuation_participants,
            std::vector<int> follower_ranks = {1})
        {
            std::vector<
                MoEOverlayInferenceCoordinatorGraphPlan::Segment>
                segments;
            segments.reserve(1u + follower_ranks.size());
            segments.push_back({
                .role =
                    MoEOverlayInferenceCoordinatorSegmentRole::Continuation,
                .materialization =
                    MoEOverlayInferenceSegmentMaterializationKind::
                        NativeDeviceExecutable,
                .world_rank = 0,
                .local_participant_count =
                    static_cast<std::size_t>(continuation_participants),
            });
            for (const int follower_rank : follower_ranks)
            {
                segments.push_back({
                    .role = MoEOverlayInferenceCoordinatorSegmentRole::
                        ExpertFollower,
                    .materialization =
                        MoEOverlayInferenceSegmentMaterializationKind::
                            EagerHostGraph,
                    .world_rank = follower_rank,
                    .local_participant_count = 1u,
                });
            }
            return MoEOverlayInferenceCoordinatorGraphPlan::
                sealAfterSynchronizedMaterialization(
                    /*graph_family_generation=*/71u,
                    std::move(segments));
        }

        /** @brief Restore one process environment value after a focused test. */
        class ScopedEnvironmentVariable final
        {
        public:
            /** @brief Set @p name to @p value for this object's lifetime. */
            ScopedEnvironmentVariable(const char *name, const char *value)
                : name_(name),
                  had_old_value_(std::getenv(name) != nullptr),
                  old_value_(had_old_value_ ? std::getenv(name) : "")
            {
                setenv(name_.c_str(), value, 1);
            }

            /** @brief Restore the exact prior environment state. */
            ~ScopedEnvironmentVariable()
            {
                if (had_old_value_)
                    setenv(name_.c_str(), old_value_.c_str(), 1);
                else
                    unsetenv(name_.c_str());
            }

            ScopedEnvironmentVariable(
                const ScopedEnvironmentVariable &) = delete;
            ScopedEnvironmentVariable &operator=(
                const ScopedEnvironmentVariable &) = delete;

        private:
            std::string name_;
            bool had_old_value_ = false;
            std::string old_value_;
        };

        /** @brief Device/MPI-free publisher used to test rank-wide coordination. */
        class RecordingPublisher final
            : public IMoEOverlayInferenceTransactionPublisher
        {
        public:
            explicit RecordingPublisher(int target_rank)
                : topology_(topology(target_rank))
            {
            }

            bool beginCommand(
                const MoEOverlayInferenceCommandIdentity &command_identity,
                std::string *error) override
            {
                if (error)
                    error->clear();
                if (!command_identity.valid() || active_)
                    return false;
                command_ = command_identity;
                active_ = true;
                next_ordinal_ = 1;
                return true;
            }

            MoEOverlayPublishedInferenceTransaction publish(
                const MoEOverlayInferenceExecutionDescriptor &descriptor)
                override
            {
                MoEOverlayPublishedInferenceTransaction result;
                if (!active_)
                {
                    result.error = "recording publisher is inactive";
                    return result;
                }
                try
                {
                    result.ticket = makeMoEOverlayInferenceExecutionTicket(
                        topology_, command_, next_ordinal_++,
                        descriptor.logical_step_id,
                        descriptor.placement_epoch,
                        descriptor.graph_role,
                        descriptor.request_count,
                        descriptor.logical_rows_per_request,
                        descriptor.physical_rows_per_request,
                        descriptor.draft_depth,
                        descriptor.sidecar_depth,
                        descriptor.prefill_schedule_workload);
                }
                catch (const std::exception &exception)
                {
                    result.error = exception.what();
                    return result;
                }
                result.ok = true;
                result.slot_index = publish_count_++;
                tickets_.push_back(result.ticket);
                return result;
            }

            bool retire(
                const MoEOverlayPublishedInferenceTransaction &transaction,
                std::string *error) override
            {
                if (error)
                    error->clear();
                if (!active_ || !transaction.ok)
                    return false;
                ++retire_count_;
                return true;
            }

            bool complete(
                std::uint64_t placement_epoch,
                std::string *error) override
            {
                if (error)
                    error->clear();
                if (!active_ || placement_epoch == 0)
                    return false;
                active_ = false;
                ++complete_count_;
                return true;
            }

            bool abort(
                std::uint64_t placement_epoch,
                int error_code,
                std::string *error) override
            {
                if (error)
                    error->clear();
                if (!active_ || placement_epoch == 0 || error_code <= 0)
                    return false;
                active_ = false;
                ++abort_count_;
                return true;
            }

            const MoEOverlayInferenceTopologyIdentity &topologyIdentity()
                const noexcept override
            {
                return topology_;
            }

            std::size_t publishCount() const noexcept { return publish_count_; }
            std::size_t retireCount() const noexcept { return retire_count_; }
            std::size_t completeCount() const noexcept { return complete_count_; }
            std::size_t abortCount() const noexcept { return abort_count_; }
            const std::vector<MoEOverlayInferenceTransactionTicket> &tickets()
                const noexcept
            {
                return tickets_;
            }

        private:
            MoEOverlayInferenceTopologyIdentity topology_;
            MoEOverlayInferenceCommandIdentity command_{};
            bool active_ = false;
            std::uint64_t next_ordinal_ = 1;
            std::size_t publish_count_ = 0;
            std::size_t retire_count_ = 0;
            std::size_t complete_count_ = 0;
            std::size_t abort_count_ = 0;
            std::vector<MoEOverlayInferenceTransactionTicket> tickets_;
        };

        /** @brief Controllable non-blocking device terminal for coordinator tests. */
        class RecordingCompletionFence final
            : public IMoEOverlayInferenceCompletionEvent
        {
        public:
            /** @copydoc IMoEOverlayInferenceCompletionEvent::record */
            bool record(
                void *producer_stream,
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (!producer_stream || recorded_.exchange(true))
                    return false;
                return true;
            }

            /** @copydoc IMoEOverlayInferenceCompletionFence::poll */
            [[nodiscard]] MoEOverlayInferenceCompletionFenceProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (!recorded_.load(std::memory_order_acquire))
                    return MoEOverlayInferenceCompletionFenceProgress::Failed;
                return ready_.load(std::memory_order_acquire)
                           ? MoEOverlayInferenceCompletionFenceProgress::Ready
                           : MoEOverlayInferenceCompletionFenceProgress::Pending;
            }

            /** @brief Publish the synthetic GPU terminal. */
            void signal() noexcept
            {
                ready_.store(true, std::memory_order_release);
            }

        private:
            std::atomic<bool> recorded_{false};
            std::atomic<bool> ready_{false};
        };

        MoEOverlayInferenceCommandIdentity command(
            std::uint64_t request_generation = 9,
            std::uint64_t command_id = 3,
            std::uint64_t placement_epoch = 41)
        {
            return {
                .request_generation = request_generation,
                .command_id = command_id,
                .initial_placement_epoch = placement_epoch,
            };
        }

        MoEOverlayInferenceTransactionProtocol protocol(
            std::size_t slot_count = 3)
        {
            return MoEOverlayInferenceTransactionProtocol({
                .topology = topology(),
                .slot_count = slot_count,
                .max_request_count = 4,
                .max_rows_per_request = 16,
                .max_mtp_draft_depth = 3,
            });
        }

        /** @brief Complete two-domain plan used by host-RCU protocol tests. */
        MoERoutedExpertPlacementPlan heterogeneousPlan()
        {
            RoutedExpertDomain continuation;
            continuation.name = "continuation";
            continuation.scope = ExecutionDomainScope::GLOBAL;
            continuation.backend = CollectiveBackendType::NCCL;
            continuation.participants = {
                GlobalDeviceAddress::cuda(0, 0, "node")};
            continuation.world_ranks = {0};
            continuation.owner_rank = 0;
            continuation.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            RoutedExpertDomain follower;
            follower.name = "follower";
            follower.scope = ExecutionDomainScope::GLOBAL;
            follower.backend = CollectiveBackendType::RCCL;
            follower.participants = {
                GlobalDeviceAddress::rocm(0, 1, "node")};
            follower.world_ranks = {1};
            follower.owner_rank = 1;
            follower.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;

            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = continuation.name;
            plan.shared_expert_domain = continuation.name;
            plan.residency_policy =
                RoutedExpertResidencyPolicy::StaticById;
            plan.domains = {continuation, follower};
            plan.routed_tiers = {
                RoutedExpertTier{.name = "priority-0",
                                 .domain = continuation.name,
                                 .priority = 0},
                RoutedExpertTier{.name = "priority-1",
                                 .domain = follower.name,
                                 .priority = 1,
                                 .fallback = true},
            };
            for (int layer = 0; layer < 3; ++layer)
            {
                plan.placements.push_back({
                    .layer = layer,
                    .routed_expert_tier = {0, 0, 1, 1},
                });
            }
            return plan;
        }

        MoEExpertOwnerMap heterogeneousOwnerMap()
        {
            return MoEExpertOwnerMap::build(heterogeneousPlan());
        }

        /** @brief Metadata matching @ref heterogeneousPlan. */
        MoERoutedExpertModelMetadata heterogeneousMetadata()
        {
            MoERoutedExpertModelMetadata metadata;
            metadata.num_layers = 3;
            metadata.main_inference_layer_count = 2;
            metadata.num_experts = 4;
            metadata.d_model = 16;
            metadata.routed_intermediate_size = 8;
            metadata.routed_quant_type = "F32";
            return metadata;
        }

        void completeSlot(
            MoEOverlayInferenceTransactionProtocol &owner,
            const MoEOverlayInferenceAdmission &admission,
            const MoEOverlayInferenceTransactionTicket &ticket)
        {
            ASSERT_TRUE(admission.accepted()) << admission.error;
            ASSERT_TRUE(owner.markSubmitted(admission.slot_index, ticket));
            ASSERT_TRUE(owner.markReturnReady(admission.slot_index, ticket));
            ASSERT_TRUE(owner.retire(admission.slot_index, ticket));
        }
    } // namespace

    TEST(Test__MoEOverlayInferenceTransaction,
         SerialDecodeOwnsSlotUntilExactReturnRetirement)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));

        const auto ticket = makeMoEOverlayInferenceExecutionTicket(
            topology(),
            active,
            1,
            73,
            41,
            MoEOverlayInferenceGraphRole::MainDecode,
            1,
            1,
            1);
        EXPECT_EQ(ticket.logical_step_id, 73u);
        const auto admission = owner.accept(ticket);
        ASSERT_TRUE(admission.accepted()) << admission.error;
        EXPECT_EQ(admission.slot_index, 0u);
        EXPECT_EQ(owner.inFlightSlotCount(), 1u);
        EXPECT_EQ(
            owner.slotState(0),
            MoEOverlayInferenceTransactionSlotState::Accepted);

        EXPECT_FALSE(owner.markReturnReady(0, ticket))
            << "A return cannot become visible before graph submission";
        ASSERT_TRUE(owner.markSubmitted(0, ticket));
        EXPECT_FALSE(owner.retire(0, ticket))
            << "An MPI slot remains owned until its return is ready";
        ASSERT_TRUE(owner.markReturnReady(0, ticket));
        ASSERT_TRUE(owner.retire(0, ticket));
        EXPECT_EQ(owner.inFlightSlotCount(), 0u);
        EXPECT_EQ(
            owner.slotState(0),
            MoEOverlayInferenceTransactionSlotState::Available);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         TopologyFingerprintCoversParticipantsAndRetainedGraphGeometry)
    {
        const auto owner_map = heterogeneousOwnerMap();
        const MoEOverlayInferenceGraphFamilyIdentity graph_family{
            .graph_family_generation = 7,
            .main_layer_count = 2,
            .mtp_source_layers = {2},
            .max_graph_rows = 16,
            .max_decode_rows = 4,
            .max_request_count = 1,
            .max_mtp_draft_depth = 3,
        };
        ASSERT_TRUE(graph_family.valid());

        const auto first = makeMoEOverlayInferenceTopologyIdentity(
            owner_map, graph_family, 0, 1);
        const auto repeated = makeMoEOverlayInferenceTopologyIdentity(
            owner_map, graph_family, 0, 1);
        EXPECT_TRUE(first.valid());
        EXPECT_EQ(first, repeated);
        EXPECT_EQ(first.workspace_generation, 7u);

        auto changed_capacity = graph_family;
        changed_capacity.max_decode_rows = 3;
        const auto changed = makeMoEOverlayInferenceTopologyIdentity(
            owner_map, changed_capacity, 0, 1);
        EXPECT_NE(first.topology_fingerprint_low,
                  changed.topology_fingerprint_low);
        EXPECT_NE(first.topology_fingerprint_high,
                  changed.topology_fingerprint_high);

        EXPECT_THROW(
            (void)makeMoEOverlayInferenceTopologyIdentity(
                owner_map, graph_family, 0, 2),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         CoordinatorGraphPlanSealsOneDeterministicRankTransactionPerSegment)
    {
        const auto plan = coordinatorGraphPlan(
            /*continuation_participants=*/2,
            /*follower_ranks=*/{3, 1});
        ASSERT_TRUE(plan.valid());
        EXPECT_EQ(plan.graphFamilyGeneration(), 71u);
        EXPECT_EQ(plan.segmentCount(), 3u);
        EXPECT_EQ(plan.followerSegmentCount(), 2u);
        EXPECT_EQ(plan.nativeSegmentCount(), 1u);
        EXPECT_EQ(plan.eagerHostSegmentCount(), 2u);
        EXPECT_EQ(plan.nativeParticipantCount(), 2u);
        EXPECT_EQ(plan.continuationWorldRank(), 0);
        ASSERT_EQ(plan.segments().size(), 3u);
        EXPECT_EQ(
            plan.segments()[0].role,
            MoEOverlayInferenceCoordinatorSegmentRole::Continuation);
        EXPECT_EQ(plan.segments()[1].world_rank, 1);
        EXPECT_EQ(plan.segments()[2].world_rank, 3);

        EXPECT_THROW(
            (void)MoEOverlayInferenceCoordinatorGraphPlan::
                sealAfterSynchronizedMaterialization(
                    71u,
                    {{.role = MoEOverlayInferenceCoordinatorSegmentRole::
                                  Continuation,
                      .materialization =
                          MoEOverlayInferenceSegmentMaterializationKind::
                              NativeDeviceExecutable,
                      .world_rank = 0,
                      .local_participant_count = 1u}}),
            std::invalid_argument)
            << "A plan without a remote transaction boundary is incomplete";
        EXPECT_THROW(
            (void)MoEOverlayInferenceCoordinatorGraphPlan::
                sealAfterSynchronizedMaterialization(
                    71u,
                    {{.role = MoEOverlayInferenceCoordinatorSegmentRole::
                                  Continuation,
                      .materialization =
                          MoEOverlayInferenceSegmentMaterializationKind::
                              NativeDeviceExecutable,
                      .world_rank = 0,
                      .local_participant_count = 1u},
                     {.role = MoEOverlayInferenceCoordinatorSegmentRole::
                                  ExpertFollower,
                      .materialization =
                          MoEOverlayInferenceSegmentMaterializationKind::
                              EagerHostGraph,
                      .world_rank = 0,
                      .local_participant_count = 1u}}),
            std::invalid_argument)
            << "Two segment roles may never alias one MPI rank";
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         CoordinatorGraphPlanMirrorsSetupAndExactSparseReturnRetirement)
    {
        ScopedEnvironmentVariable perf_export(
            "LLAMINAR_PERF_STATS_JSON", "1");
        ScopedEnvironmentVariable perf_filter(
            "LLAMINAR_PERF_STATS_FILTER", "forward_graph");
        PerfStatsCollector::reset();
        auto publisher = std::make_shared<RecordingPublisher>(1);
        MoEOverlayInferenceTransactionCoordinator coordinator(
            MoEOverlayInferenceTransactionCoordinator::Config{
                .publishers = {publisher},
                .graph_plan = coordinatorGraphPlan(1),
                .continuation_participant_count = 1,
                .ticket_authority_participant_index = 0,
                .participant_completion_boundaries = {
                    MoEOverlayInferenceCompletionBoundaryKind::
                        HostSynchronous,
                },
                .max_transactions_per_command = 1,
                .max_mtp_draft_depth = 0,
            });
        ASSERT_TRUE(coordinator.beginCommand(command()));
        ASSERT_TRUE(coordinator.beginGraphSequence(/*draft_depth=*/0));
        const auto binding = coordinator.beginParticipantGraph(
            MoEOverlayInferenceExecutionDescriptor{
                .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = 1,
                .physical_rows_per_request = 1,
            },
            /*participant_index=*/0);
        ASSERT_TRUE(binding.ok) << binding.error;
        ASSERT_TRUE(coordinator.armParticipantGraph(binding));
        ASSERT_TRUE(coordinator.finishParticipantGraph(binding, true));

        auto before_retirement =
            PerfStatsCollector::snapshot({"forward_graph"});
        EXPECT_EQ(
            std::count_if(
                before_retirement.begin(),
                before_retirement.end(),
                [](const PerfStatRecord &record)
                {
                    return record.name == "segmented_replay_segments";
                }),
            0)
            << "graph submission is not the exact sparse-return retirement";

        ASSERT_TRUE(coordinator.retireCompletedGraphSequence());
        const auto records =
            PerfStatsCollector::snapshot({"forward_graph"});
        const auto value_for = [&](const char *name)
        {
            const auto found = std::find_if(
                records.begin(),
                records.end(),
                [name](const PerfStatRecord &record)
                {
                    return record.name == name;
                });
            EXPECT_NE(found, records.end()) << name;
            return found == records.end() ? 0.0 : found->value;
        };
        EXPECT_DOUBLE_EQ(value_for("segmented_plan_segments"), 2.0);
        EXPECT_DOUBLE_EQ(
            value_for("segmented_graph_capture_segments"), 1.0);
        EXPECT_DOUBLE_EQ(value_for("segmented_replay_segments"), 2.0);
        PerfStatsCollector::reset();
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorPublishesOnceAcrossSymmetricParticipantsAndTargets)
    {
        auto first_target = std::make_shared<RecordingPublisher>(1);
        auto second_target = std::make_shared<RecordingPublisher>(2);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {first_target, second_target},
                    .graph_plan = coordinatorGraphPlan(2, {1, 2}),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(2));

        const MoEOverlayInferenceExecutionDescriptor sidecar{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        auto first = coordinator->beginParticipantGraph(sidecar, 0);
        ASSERT_TRUE(first.ok) << first.error;
        ASSERT_TRUE(first.active);
        EXPECT_EQ(first.descriptor.draft_depth, 2);
        EXPECT_EQ(first.descriptor.sidecar_depth, 0);
        EXPECT_EQ(first_target->publishCount(), 0u)
            << "Graph entry reserves identity but must not wake a follower during cold capture";
        EXPECT_EQ(second_target->publishCount(), 0u);

        auto second = coordinator->beginParticipantGraph(sidecar, 1);
        ASSERT_TRUE(second.ok) << second.error;
        EXPECT_EQ(second.group_id, first.group_id);
        EXPECT_EQ(second.descriptor, first.descriptor);
        ASSERT_TRUE(coordinator->armParticipantGraph(first));
        EXPECT_EQ(first_target->publishCount(), 1u)
            << "The first executable launch edge publishes one target ticket";
        EXPECT_EQ(second_target->publishCount(), 1u);
        EXPECT_EQ(first_target->publishCount(), 1u)
            << "A LocalTP sibling must not own another ticket edge";
        ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));

        auto third = coordinator->beginParticipantGraph(sidecar, 0);
        auto fourth = coordinator->beginParticipantGraph(sidecar, 1);
        ASSERT_TRUE(third.ok) << third.error;
        ASSERT_TRUE(fourth.ok) << fourth.error;
        EXPECT_EQ(third.descriptor.sidecar_depth, 1);
        ASSERT_TRUE(coordinator->armParticipantGraph(third));
        ASSERT_TRUE(coordinator->finishParticipantGraph(third, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(fourth, true));

        const MoEOverlayInferenceExecutionDescriptor verifier{
            .graph_role =
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 3,
            .physical_rows_per_request = 4,
            .draft_depth = 2,
        };
        auto verifier_first =
            coordinator->beginParticipantGraph(verifier, 0);
        auto verifier_second =
            coordinator->beginParticipantGraph(verifier, 1);
        ASSERT_TRUE(verifier_first.ok) << verifier_first.error;
        ASSERT_TRUE(verifier_second.ok) << verifier_second.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(verifier_first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(
            verifier_first, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(
            verifier_second, true));

        EXPECT_EQ(first_target->retireCount(), 0u)
            << "Asynchronous graph slots remain live until terminal output";
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(first_target->publishCount(), 3u);
        EXPECT_EQ(second_target->publishCount(), 3u);
        EXPECT_EQ(first_target->retireCount(), 3u);
        EXPECT_EQ(second_target->retireCount(), 3u);
        EXPECT_EQ(first_target->completeCount(), 1u);
        EXPECT_EQ(second_target->completeCount(), 1u);
    }

    /**
     * @brief A numeric sequence epoch must own its host-RCU lifetime.
     *
     * Maintenance may publish a successor between sparse layers or retained
     * graph segments. The coordinator therefore holds one authority lease from
     * sequence admission through exact sparse-return retirement; relying only
     * on the short per-layer dispatch lease would let the old bank disappear
     * before the next layer reacquired the same epoch.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorPinsHostResidencyForCompleteGraphSequence)
    {
        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = heterogeneousPlan(),
                .model_metadata = heterogeneousMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
            });
        auto publisher = std::make_shared<RecordingPublisher>(1);
        MoEOverlayInferenceTransactionCoordinator coordinator(
            MoEOverlayInferenceTransactionCoordinator::Config{
                .publishers = {publisher},
                .graph_plan = coordinatorGraphPlan(1),
                .continuation_participant_count = 1,
                .ticket_authority_participant_index = 0,
                .participant_completion_boundaries = {
                    MoEOverlayInferenceCompletionBoundaryKind::
                        HostSynchronous,
                },
                .max_transactions_per_command = 2,
                .max_mtp_draft_depth = 0,
                .residency_authority = authority,
            });

        ASSERT_TRUE(coordinator.beginCommand(command(
            /*request_generation=*/9,
            /*command_id=*/3,
            /*placement_epoch=*/1)));
        EXPECT_EQ(authority->activeTicketCount(), 0u);
        ASSERT_TRUE(coordinator.beginGraphSequence(/*draft_depth=*/0));
        EXPECT_EQ(authority->activeTicketCount(), 1u)
            << "the complete sequence, not an individual layer, owns the RCU lease";
        EXPECT_EQ(coordinator.currentPlacementEpoch(), 1u);

        const MoEOverlayInferenceExecutionDescriptor decode{
            .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
            .placement_epoch = coordinator.currentPlacementEpoch(),
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        const auto binding = coordinator.beginParticipantGraph(decode, 0);
        ASSERT_TRUE(binding.ok) << binding.error;
        ASSERT_TRUE(coordinator.armParticipantGraph(binding));
        ASSERT_TRUE(coordinator.finishParticipantGraph(binding, true));
        EXPECT_EQ(authority->activeTicketCount(), 1u)
            << "graph submission alone is not the sparse-return retirement edge";

        ASSERT_TRUE(coordinator.completeCommand(1));
        EXPECT_EQ(authority->activeTicketCount(), 0u)
            << "the exact command terminal retires the final sequence lease";
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorRejectsDivergentParticipantGeometry)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 3,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(0));
        const MoEOverlayInferenceExecutionDescriptor decode{
            .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        auto first = coordinator->beginParticipantGraph(decode, 0);
        ASSERT_TRUE(first.ok) << first.error;

        auto divergent = decode;
        divergent.physical_rows_per_request = 2;
        auto second = coordinator->beginParticipantGraph(divergent, 1);
        EXPECT_FALSE(second.ok);
        EXPECT_NE(second.error.find("divergent"), std::string::npos);
        EXPECT_EQ(publisher->publishCount(), 0u)
            << "Divergent LocalTP admission must fail before any follower is armed";
        EXPECT_TRUE(coordinator->abortCommand(41, 7));
        EXPECT_EQ(publisher->abortCount(), 1u);
    }

    /**
     * @brief Ticket authority follows planner topology instead of child zero.
     *
     * A passive LocalTP sibling may complete its local graph before the packet
     * root reaches the retained-parent launch edge. The coordinator accepts that
     * ordering, but the graph group cannot complete successfully until the
     * explicitly configured authority publishes the one remote ticket.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorUsesConfiguredNonzeroTicketAuthority)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 1,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 0,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(0));

        const MoEOverlayInferenceExecutionDescriptor decode{
            .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        const auto sibling =
            coordinator->beginParticipantGraph(decode, 0);
        const auto authority =
            coordinator->beginParticipantGraph(decode, 1);
        ASSERT_TRUE(sibling.ok) << sibling.error;
        ASSERT_TRUE(authority.ok) << authority.error;
        EXPECT_FALSE(sibling.owns_ticket_authority);
        EXPECT_TRUE(authority.owns_ticket_authority);

        ASSERT_TRUE(coordinator->finishParticipantGraph(sibling, true));
        EXPECT_EQ(publisher->publishCount(), 0u);
        ASSERT_TRUE(coordinator->armParticipantGraph(authority));
        EXPECT_EQ(publisher->publishCount(), 1u);
        ASSERT_TRUE(coordinator->finishParticipantGraph(authority, true));
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->retireCount(), 1u);
        EXPECT_EQ(publisher->completeCount(), 1u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorRollsBoundedSerialPrefillChunksWithoutDuplicateTickets)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        std::uint64_t retired_prefill_tokens = 0u;
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
                    .retired_prefill_progress_sink =
                        [&retired_prefill_tokens](
                            std::uint64_t completed_tokens,
                            std::string *)
                    {
                        retired_prefill_tokens += completed_tokens;
                        return true;
                    },
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));

        const MoEOverlayInferenceExecutionDescriptor prefill{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 8,
            .physical_rows_per_request = 8,
        };
        auto submit_chunk = [&]
        {
            ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
            auto first = coordinator->beginParticipantGraph(prefill, 0);
            ASSERT_TRUE(first.ok) << first.error;
            ASSERT_TRUE(first.active);
            ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
            auto second = coordinator->beginParticipantGraph(prefill, 1);
            ASSERT_TRUE(second.ok) << second.error;
            EXPECT_EQ(second.group_id, first.group_id);
            EXPECT_EQ(second.descriptor, first.descriptor);
            ASSERT_TRUE(coordinator->armParticipantGraph(first));
            ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
            ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        };

        submit_chunk();
        EXPECT_EQ(publisher->publishCount(), 1u);
        EXPECT_EQ(publisher->retireCount(), 0u);
        EXPECT_EQ(retired_prefill_tokens, 0u)
            << "Progress becomes visible only after the sparse return retires";

        submit_chunk();
        EXPECT_EQ(publisher->publishCount(), 2u);
        EXPECT_EQ(publisher->retireCount(), 1u)
            << "The first complete chunk must release its fixed protocol slot "
               "before the next chunk publishes";
        EXPECT_EQ(retired_prefill_tokens, 8u);

        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->retireCount(), 2u);
        EXPECT_EQ(publisher->completeCount(), 1u);
        EXPECT_EQ(retired_prefill_tokens, 16u)
            << "Each real prefill row must advance cadence exactly once";
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorPublishesOneExactIntervalPerPrefillGraphGroup)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 0,
                });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        ASSERT_TRUE(coordinator->bindPrefillInterferenceProbe(probe));

        const MoEOverlayInterferenceProbeRequest request{
            .coordinate = {
                .source_participant = 0,
                .destination_participant = 1,
                .layer = 0,
            },
            .source = ExpertHistogramSource::PrefillChunk,
            .mode = MoEOverlayInterferenceProbeMode::Baseline,
            .calibration_sequence = 1,
        };
        ASSERT_TRUE(probe->arm(request));
        ASSERT_TRUE(coordinator->beginCommand(command()));

        const MoEOverlayInferenceExecutionDescriptor tail_chunk{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 3,
            .physical_rows_per_request = 8,
        };
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
        const auto first =
            coordinator->beginParticipantGraph(tail_chunk, 0);
        ASSERT_TRUE(first.ok) << first.error;
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "The first symmetric entrant owns the start of the rank-wide interval";

        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
        const auto second =
            coordinator->beginParticipantGraph(tail_chunk, 1);
        ASSERT_TRUE(second.ok) << second.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "A fast sibling cannot truncate the slower participant's graph";
        ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Completed);

        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe->consume(&sample));
        EXPECT_EQ(
            sample.workload,
            makeMoEOverlayInferenceWorkloadIdentity(
                ExpertHistogramSource::PrefillChunk,
                /*real_rows=*/3,
                /*execution_rows=*/8,
                /*transaction_count=*/1,
                /*speculative_depth=*/0));
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_TRUE(probe->idle());
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         GpuCoordinatorWaitsForEveryExactDeviceTerminalBeforeCompletingProbe)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent,
                        MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 0,
                });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        ASSERT_TRUE(coordinator->bindPrefillInterferenceProbe(probe));
        const MoEOverlayInterferenceProbeRequest request{
            .coordinate = {
                .source_participant = 0,
                .destination_participant = 1,
                .layer = 0,
            },
            .source = ExpertHistogramSource::PrefillChunk,
            .mode = MoEOverlayInterferenceProbeMode::Baseline,
            .calibration_sequence = 13,
        };
        ASSERT_TRUE(probe->arm(request));
        ASSERT_TRUE(coordinator->beginCommand(command()));

        const MoEOverlayInferenceExecutionDescriptor chunk{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 64,
            .physical_rows_per_request = 64,
        };
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
        const auto authority = coordinator->beginParticipantGraph(chunk, 0);
        ASSERT_TRUE(authority.ok) << authority.error;
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
        const auto sibling = coordinator->beginParticipantGraph(chunk, 1);
        ASSERT_TRUE(sibling.ok) << sibling.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(authority));

        auto authority_fence =
            std::make_shared<RecordingCompletionFence>();
        auto sibling_fence =
            std::make_shared<RecordingCompletionFence>();
        std::string error;
        ASSERT_TRUE(
            coordinator->deferPrefillInterferenceCompletionAtDeviceTerminal(
                authority.descriptor.logical_step_id,
                authority.participant_index,
                authority_fence,
                reinterpret_cast<void *>(0x1),
                &error)) << error;
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "One participant terminal must not arm a rank-wide sample";
        ASSERT_TRUE(
            coordinator->deferPrefillInterferenceCompletionAtDeviceTerminal(
                sibling.descriptor.logical_step_id,
                sibling.participant_index,
                sibling_fence,
                reinterpret_cast<void *>(0x2),
                &error)) << error;
        ASSERT_TRUE(coordinator->finishParticipantGraph(authority, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(sibling, true));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "Host graph submission must not close the GPU interval";

        authority_fence->signal();
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "A ready authority event must remain idempotent while its sibling is pending";
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running);

        sibling_fence->signal();
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Completed);
        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe->consume(&sample));
        EXPECT_EQ(sample.workload.real_rows, 64);
        ASSERT_TRUE(coordinator->completeCommand(41));
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         MixedCoordinatorCombinesGpuEventAndCpuSynchronousTerminal)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 0,
                });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        ASSERT_TRUE(coordinator->bindPrefillInterferenceProbe(probe));
        const MoEOverlayInterferenceProbeRequest request{
            .coordinate = {
                .source_participant = 0,
                .destination_participant = 1,
                .layer = 0,
            },
            .source = ExpertHistogramSource::PrefillChunk,
            .mode = MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            .calibration_sequence = 17,
        };
        ASSERT_TRUE(probe->arm(request));
        ASSERT_TRUE(coordinator->beginCommand(command()));

        const MoEOverlayInferenceExecutionDescriptor chunk{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 32,
            .physical_rows_per_request = 64,
        };
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
        const auto gpu = coordinator->beginParticipantGraph(chunk, 0);
        ASSERT_TRUE(gpu.ok) << gpu.error;
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
        const auto cpu = coordinator->beginParticipantGraph(chunk, 1);
        ASSERT_TRUE(cpu.ok) << cpu.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(gpu));

        auto gpu_event = std::make_shared<RecordingCompletionFence>();
        std::string error;
        ASSERT_TRUE(
            coordinator->deferPrefillInterferenceCompletionAtDeviceTerminal(
                gpu.descriptor.logical_step_id,
                gpu.participant_index,
                gpu_event,
                reinterpret_cast<void *>(0x3),
                &error)) << error;
        ASSERT_TRUE(coordinator->finishParticipantGraph(gpu, true));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "GPU submission alone must not stand in for the CPU call terminal";
        ASSERT_TRUE(coordinator->finishParticipantGraph(cpu, true));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "CPU completion arms the aggregate but cannot truncate GPU work";

        gpu_event->signal();
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Completed);
        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe->consume(&sample));
        EXPECT_EQ(sample.workload.real_rows, 32);
        EXPECT_EQ(sample.workload.execution_rows, 64);
        ASSERT_TRUE(coordinator->completeCommand(41));
    }

    /**
     * A migration may outlive one retained prefill segment while remaining
     * wholly inside the caller-visible bucket schedule.  This regression proves
     * that the coordinator claims the complete workload once, stamps it into
     * every authenticated follower ticket, and closes it only at the final
     * segment's exact device events.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         SegmentedPrefillCalibrationOwnsOneAggregateDeviceTerminal)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent,
                        MoEOverlayInferenceCompletionBoundaryKind::DeviceEvent,
                    },
                    .max_transactions_per_command = 2,
                    .max_mtp_draft_depth = 0,
                });
        auto probe =
            std::make_shared<MoEOverlayInferenceInterferenceProbe>();
        ASSERT_TRUE(coordinator->bindPrefillInterferenceProbe(probe));

        const auto workload = makeMoEOverlayInferenceWorkloadIdentity(
            ExpertHistogramSource::PrefillChunk,
            /*real_rows=*/7,
            /*execution_rows=*/16,
            /*transaction_count=*/2,
            /*speculative_depth=*/0,
            /*schedule_fingerprint=*/0x123456789abcdef0ULL);
        const MoEOverlayInterferenceProbeRequest request{
            .coordinate = {
                .source_participant = 0,
                .destination_participant = 1,
                .layer = 0,
            },
            .source = ExpertHistogramSource::PrefillChunk,
            .require_exact_workload = true,
            .required_workload = workload,
            .mode = MoEOverlayInterferenceProbeMode::ConcurrentMovement,
            .calibration_sequence = 29,
        };
        ASSERT_TRUE(probe->arm(request));
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(
            coordinator->declarePrefillInterferenceSchedule(workload));
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running);

        auto submit = [&](int real_rows,
                          std::shared_ptr<RecordingCompletionFence> first_event,
                          std::shared_ptr<RecordingCompletionFence> second_event)
        {
            const MoEOverlayInferenceExecutionDescriptor chunk{
                .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = real_rows,
                .physical_rows_per_request = 8,
            };
            ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
            const auto first =
                coordinator->beginParticipantGraph(chunk, 0);
            ASSERT_TRUE(first.ok) << first.error;
            ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
            const auto second =
                coordinator->beginParticipantGraph(chunk, 1);
            ASSERT_TRUE(second.ok) << second.error;
            EXPECT_EQ(first.descriptor.prefill_schedule_workload, workload);
            EXPECT_EQ(second.descriptor.prefill_schedule_workload, workload);
            ASSERT_TRUE(coordinator->armParticipantGraph(first));

            std::string error;
            ASSERT_TRUE(
                coordinator->deferPrefillInterferenceCompletionAtDeviceTerminal(
                    first.descriptor.logical_step_id,
                    first.participant_index,
                    std::move(first_event),
                    reinterpret_cast<void *>(0x11),
                    &error)) << error;
            ASSERT_TRUE(
                coordinator->deferPrefillInterferenceCompletionAtDeviceTerminal(
                    second.descriptor.logical_step_id,
                    second.participant_index,
                    std::move(second_event),
                    reinterpret_cast<void *>(0x12),
                    &error)) << error;
            ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
            ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        };

        auto unused_first = std::make_shared<RecordingCompletionFence>();
        auto unused_second = std::make_shared<RecordingCompletionFence>();
        submit(4, unused_first, unused_second);
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running)
            << "A non-final segment must not publish or query device events";

        auto final_first = std::make_shared<RecordingCompletionFence>();
        auto final_second = std::make_shared<RecordingCompletionFence>();
        submit(3, final_first, final_second);
        ASSERT_EQ(publisher->tickets().size(), 2u);
        for (const auto &ticket : publisher->tickets())
            EXPECT_EQ(ticket.prefillScheduleWorkload(), workload);
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running);

        final_first->signal();
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Running);
        final_second->signal();
        EXPECT_EQ(
            probe->progress(request),
            MoEOverlayInterferenceProbeProgress::Completed);

        MoEOverlayInterferenceProbeSample sample;
        ASSERT_TRUE(probe->consume(&sample));
        EXPECT_EQ(sample.workload, workload);
        ASSERT_TRUE(coordinator->completeCommand(41));
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         AheadPrefillParticipantWaitsForSiblingBeforeEnteringNextChunk)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 0,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));

        const MoEOverlayInferenceExecutionDescriptor full_chunk{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 8,
            .physical_rows_per_request = 8,
        };
        const MoEOverlayInferenceExecutionDescriptor tail_chunk{
            .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 3,
            .physical_rows_per_request = 8,
        };

        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(0));
        const auto first =
            coordinator->beginParticipantGraph(full_chunk, 0);
        ASSERT_TRUE(first.ok) << first.error;
        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
        const auto second =
            coordinator->beginParticipantGraph(full_chunk, 1);
        ASSERT_TRUE(second.ok) << second.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(first));

        // Participant zero has submitted its asynchronous GPU graph and races
        // toward the tail while participant one is deliberately held back.
        ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
        std::promise<void> admission_started;
        auto admission_started_future = admission_started.get_future();
        auto ahead = std::async(
            std::launch::async,
            [&]
            {
                admission_started.set_value();
                std::string error;
                if (!coordinator->admitSerialPrefillGraph(0, &error))
                {
                    MoEOverlayInferenceParticipantGraphBinding failed;
                    failed.error = std::move(error);
                    return failed;
                }
                return coordinator->beginParticipantGraph(tail_chunk, 0);
            });
        admission_started_future.wait();
        EXPECT_EQ(
            ahead.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout)
            << "An ahead participant must not alias the still-active chunk.";

        ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        auto tail_first = ahead.get();
        ASSERT_TRUE(tail_first.ok) << tail_first.error;
        ASSERT_TRUE(tail_first.active);
        EXPECT_NE(tail_first.group_id, first.group_id);
        EXPECT_EQ(tail_first.descriptor.logical_rows_per_request, 3);

        ASSERT_TRUE(coordinator->admitSerialPrefillGraph(1));
        const auto tail_second =
            coordinator->beginParticipantGraph(tail_chunk, 1);
        ASSERT_TRUE(tail_second.ok) << tail_second.error;
        EXPECT_EQ(tail_second.group_id, tail_first.group_id);
        ASSERT_TRUE(coordinator->armParticipantGraph(tail_first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(tail_first, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(tail_second, true));
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->publishCount(), 2u);
        EXPECT_EQ(publisher->retireCount(), 2u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorRecyclesFixedSlotsAcrossDynamicDeviceTransactions)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));

        auto execute_sequence = [&](int depth)
        {
            ASSERT_TRUE(coordinator->beginGraphSequence(depth));
            const MoEOverlayInferenceExecutionDescriptor sidecar{
                .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = 1,
                .physical_rows_per_request = 1,
            };
            for (int ordinal = 0; ordinal < depth; ++ordinal)
            {
                auto first = coordinator->beginParticipantGraph(sidecar, 0);
                auto second = coordinator->beginParticipantGraph(sidecar, 1);
                ASSERT_TRUE(first.ok) << first.error;
                ASSERT_TRUE(second.ok) << second.error;
                EXPECT_EQ(first.descriptor.sidecar_depth, ordinal);
                ASSERT_TRUE(coordinator->armParticipantGraph(first));
                ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
                ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
            }

            const MoEOverlayInferenceExecutionDescriptor verifier{
                .graph_role =
                    MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = depth + 1,
                .physical_rows_per_request = 4,
                .draft_depth = depth,
            };
            auto first = coordinator->beginParticipantGraph(verifier, 0);
            auto second = coordinator->beginParticipantGraph(verifier, 1);
            ASSERT_TRUE(first.ok) << first.error;
            ASSERT_TRUE(second.ok) << second.error;
            ASSERT_TRUE(coordinator->armParticipantGraph(first));
            ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
            ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));

            ASSERT_TRUE(coordinator->retireCompletedGraphSequence());
            EXPECT_EQ(publisher->publishCount(), publisher->retireCount());
            EXPECT_EQ(coordinator->activeMTPDraftDepth(), -1);
        };

        execute_sequence(2);
        execute_sequence(3);
        execute_sequence(2);
        EXPECT_EQ(publisher->publishCount(), 10u);
        EXPECT_EQ(publisher->retireCount(), 10u);
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->completeCount(), 1u);
    }

    /**
     * @brief Maximum dynamic depth uses one immutable role plan and epoch.
     *
     * Depth fifteen is the production controller limit. Exercising every graph
     * ordinal prevents a fixed-slot capacity increase from silently preserving
     * the old independent sidecar counters or accepting an early verifier.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         DepthFifteenSequencePinsOneEpochAndExactRoleOrder)
    {
        constexpr int kDepth = 15;
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(1),
                    .continuation_participant_count = 1,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = kDepth + 1,
                    .max_mtp_draft_depth = kDepth,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(kDepth));

        const MoEOverlayInferenceExecutionDescriptor sidecar{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        for (int ordinal = 0; ordinal < kDepth; ++ordinal)
        {
            const auto binding =
                coordinator->beginParticipantGraph(sidecar, 0);
            ASSERT_TRUE(binding.ok) << binding.error;
            EXPECT_EQ(binding.descriptor.draft_depth, kDepth);
            EXPECT_EQ(binding.descriptor.sidecar_depth, ordinal);
            EXPECT_EQ(binding.descriptor.placement_epoch, 41u);
            ASSERT_TRUE(coordinator->armParticipantGraph(binding));
            ASSERT_TRUE(coordinator->finishParticipantGraph(binding, true));
        }

        const MoEOverlayInferenceExecutionDescriptor verifier{
            .graph_role =
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = kDepth + 1,
            .physical_rows_per_request = kDepth + 1,
            .draft_depth = kDepth,
        };
        const auto terminal =
            coordinator->beginParticipantGraph(verifier, 0);
        ASSERT_TRUE(terminal.ok) << terminal.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(terminal));
        ASSERT_TRUE(coordinator->finishParticipantGraph(terminal, true));
        ASSERT_TRUE(coordinator->retireCompletedGraphSequence());
        ASSERT_TRUE(coordinator->completeCommand(41));

        ASSERT_EQ(publisher->tickets().size(),
                  static_cast<std::size_t>(kDepth + 1));
        for (const auto &ticket : publisher->tickets())
            EXPECT_EQ(ticket.placement_epoch, 41u);
        EXPECT_EQ(publisher->publishCount(), publisher->retireCount());
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         SequenceRejectsVerifierBeforeEveryDraftGraph)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(1),
                    .continuation_participant_count = 1,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 3,
                    .max_mtp_draft_depth = 2,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(2));
        const MoEOverlayInferenceExecutionDescriptor verifier{
            .graph_role =
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 3,
            .physical_rows_per_request = 3,
            .draft_depth = 2,
        };
        const auto binding =
            coordinator->beginParticipantGraph(verifier, 0);
        EXPECT_FALSE(binding.ok);
        EXPECT_NE(binding.error.find("out of order"), std::string::npos);
        EXPECT_EQ(publisher->publishCount(), 0u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         SequenceRejectsResidencyEpochChangeBetweenDraftGraphs)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(1),
                    .continuation_participant_count = 1,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 3,
                    .max_mtp_draft_depth = 2,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(2));
        MoEOverlayInferenceExecutionDescriptor sidecar{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        const auto first = coordinator->beginParticipantGraph(sidecar, 0);
        ASSERT_TRUE(first.ok) << first.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));

        sidecar.placement_epoch = 42;
        const auto second = coordinator->beginParticipantGraph(sidecar, 0);
        EXPECT_FALSE(second.ok);
        EXPECT_NE(second.error.find("residency epoch"), std::string::npos);
        EXPECT_EQ(publisher->publishCount(), 1u);
    }

    /**
     * @brief Symmetric hosted participants share one ticket-keyed transition.
     *
     * Persistent LocalTP workers may reach the transaction boundary in either
     * order. Both must be able to present the same authenticated controller
     * ticket without electing participant zero or racing a mutable depth read.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         HostedSequenceTransitionIsIdempotentAcrossSymmetricParticipants)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));

        auto execute_active_sequence = [&](int depth)
        {
            const MoEOverlayInferenceExecutionDescriptor sidecar{
                .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = 1,
                .physical_rows_per_request = 1,
            };
            for (int ordinal = 0; ordinal < depth; ++ordinal)
            {
                auto first = coordinator->beginParticipantGraph(sidecar, 0);
                auto second = coordinator->beginParticipantGraph(sidecar, 1);
                ASSERT_TRUE(first.ok) << first.error;
                ASSERT_TRUE(second.ok) << second.error;
                ASSERT_TRUE(coordinator->armParticipantGraph(first));
                ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
                ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
            }

            const MoEOverlayInferenceExecutionDescriptor verifier{
                .graph_role =
                    MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                .placement_epoch = 41,
                .request_count = 1,
                .logical_rows_per_request = depth + 1,
                .physical_rows_per_request = 4,
                .draft_depth = depth,
            };
            auto first = coordinator->beginParticipantGraph(verifier, 0);
            auto second = coordinator->beginParticipantGraph(verifier, 1);
            ASSERT_TRUE(first.ok) << first.error;
            ASSERT_TRUE(second.ok) << second.error;
            ASSERT_TRUE(coordinator->armParticipantGraph(first));
            ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));
            ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        };

        ASSERT_TRUE(coordinator->beginGraphSequence(/*draft_depth=*/1));
        execute_active_sequence(1);

        auto first_advance = std::async(
            std::launch::async,
            [&]
            {
                return coordinator->advanceHostedGraphSequence(1, 2);
            });
        auto sibling_advance = std::async(
            std::launch::async,
            [&]
            {
                return coordinator->advanceHostedGraphSequence(1, 2);
            });
        EXPECT_TRUE(first_advance.get());
        EXPECT_TRUE(sibling_advance.get());
        EXPECT_EQ(coordinator->activeMTPDraftDepth(), 2);
        EXPECT_EQ(publisher->publishCount(), publisher->retireCount());

        execute_active_sequence(2);
        EXPECT_TRUE(coordinator->advanceHostedGraphSequence(2, std::nullopt));
        EXPECT_TRUE(coordinator->advanceHostedGraphSequence(2, std::nullopt));
        EXPECT_EQ(coordinator->activeMTPDraftDepth(), -1);
        EXPECT_EQ(publisher->publishCount(), 5u);
        EXPECT_EQ(publisher->retireCount(), 5u);
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->completeCount(), 1u);
    }

    /**
     * @brief An ahead MTP participant waits without a rank fragment barrier.
     *
     * The fast worker has already submitted its first sidecar when it asks for
     * the second. The coordinator must hold that host metadata admission until
     * the sibling seals sidecar zero, then issue the next immutable group.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         AheadMTPParticipantWaitsForSymmetricGraphSubmission)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(2),
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 2,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));
        ASSERT_TRUE(coordinator->beginGraphSequence(/*draft_depth=*/2));

        const MoEOverlayInferenceExecutionDescriptor sidecar{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        };
        auto first = coordinator->beginParticipantGraph(sidecar, 0);
        auto second = coordinator->beginParticipantGraph(sidecar, 1);
        ASSERT_TRUE(first.ok) << first.error;
        ASSERT_TRUE(second.ok) << second.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(first, true));

        auto ahead = std::async(
            std::launch::async,
            [&]
            {
                return coordinator->beginParticipantGraph(sidecar, 0);
            });
        EXPECT_EQ(
            ahead.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout)
            << "A fast participant must not alias the still-active sparse graph.";

        ASSERT_TRUE(coordinator->finishParticipantGraph(second, true));
        auto next_first = ahead.get();
        ASSERT_TRUE(next_first.ok) << next_first.error;
        EXPECT_EQ(next_first.descriptor.sidecar_depth, 1);
        auto next_second = coordinator->beginParticipantGraph(sidecar, 1);
        ASSERT_TRUE(next_second.ok) << next_second.error;
        EXPECT_EQ(next_second.group_id, next_first.group_id);
        ASSERT_TRUE(coordinator->armParticipantGraph(next_first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(next_first, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(next_second, true));

        const MoEOverlayInferenceExecutionDescriptor verifier{
            .graph_role =
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 3,
            .physical_rows_per_request = 4,
            .draft_depth = 2,
        };
        auto verifier_first =
            coordinator->beginParticipantGraph(verifier, 0);
        auto verifier_second =
            coordinator->beginParticipantGraph(verifier, 1);
        ASSERT_TRUE(verifier_first.ok) << verifier_first.error;
        ASSERT_TRUE(verifier_second.ok) << verifier_second.error;
        ASSERT_TRUE(coordinator->armParticipantGraph(verifier_first));
        ASSERT_TRUE(coordinator->finishParticipantGraph(
            verifier_first, true));
        ASSERT_TRUE(coordinator->finishParticipantGraph(
            verifier_second, true));
        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->publishCount(), 3u);
        EXPECT_EQ(publisher->retireCount(), 3u);
    }

    /**
     * @brief A condition main graph and its verifier are distinct sequences.
     *
     * Response-budget clipping is resolved only after the main condition row
     * has produced its target token. The outer command must therefore retire
     * that serial transaction before admitting the exact speculative width,
     * rather than declaring the configured maximum around both shapes.
     */
    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorRetiresSerialConditionBeforeExactSpeculativeWidth)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .graph_plan = coordinatorGraphPlan(1),
                    .continuation_participant_count = 1,
                    .ticket_authority_participant_index = 0,
                    .participant_completion_boundaries = {
                        MoEOverlayInferenceCompletionBoundaryKind::
                            HostSynchronous,
                    },
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
                });
        ASSERT_TRUE(coordinator->beginCommand(command()));

        auto execute = [&](MoEOverlayInferenceExecutionDescriptor descriptor)
        {
            auto binding = coordinator->beginParticipantGraph(descriptor, 0);
            ASSERT_TRUE(binding.ok) << binding.error;
            ASSERT_TRUE(coordinator->armParticipantGraph(binding));
            ASSERT_TRUE(coordinator->finishParticipantGraph(binding, true));
        };

        ASSERT_TRUE(coordinator->beginGraphSequence(/*draft_depth=*/0));
        execute(MoEOverlayInferenceExecutionDescriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MainDecode,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        });
        ASSERT_TRUE(coordinator->retireCompletedGraphSequence());

        ASSERT_TRUE(coordinator->beginGraphSequence(/*draft_depth=*/1));
        execute(MoEOverlayInferenceExecutionDescriptor{
            .graph_role = MoEOverlayInferenceGraphRole::MTPDraft,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 1,
            .physical_rows_per_request = 1,
        });
        execute(MoEOverlayInferenceExecutionDescriptor{
            .graph_role =
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
            .placement_epoch = 41,
            .request_count = 1,
            .logical_rows_per_request = 2,
            .physical_rows_per_request = 4,
            .draft_depth = 1,
        });

        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->publishCount(), 3u);
        EXPECT_EQ(publisher->retireCount(), 3u);
        EXPECT_EQ(publisher->completeCount(), 1u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         DepthTwoDepthThreeAndDynamicVerifierGeometryUseOneProtocol)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));

        std::uint64_t ordinal = 1;
        for (const int depth : {2, 3, 2})
        {
            for (int sidecar = 0; sidecar < depth; ++sidecar)
            {
                const auto draft = makeMoEOverlayInferenceExecutionTicket(
                    topology(),
                    active,
                    ordinal++,
                    50,
                    41,
                    MoEOverlayInferenceGraphRole::MTPDraft,
                    1,
                    1,
                    1,
                    depth,
                    sidecar);
                const auto admitted = owner.accept(draft);
                completeSlot(owner, admitted, draft);
            }

            const auto verifier = makeMoEOverlayInferenceExecutionTicket(
                topology(),
                active,
                ordinal++,
                50,
                41,
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                1,
                depth + 1,
                4,
                depth);
            const auto admitted = owner.accept(verifier);
            completeSlot(owner, admitted, verifier);
        }

        const auto terminal = makeMoEOverlayInferenceTerminalTicket(
            topology(),
            active,
            ordinal,
            41,
            MoEOverlayInferenceTransactionAction::Complete);
        EXPECT_TRUE(owner.accept(terminal).completed());
        EXPECT_EQ(owner.state(), MoEOverlayInferenceProtocolState::Complete);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         WrappedRingReportsBackpressureInsteadOfOverwritingLiveStorage)
    {
        auto owner = protocol(2);
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));

        const auto first = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 1, 10, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        const auto second = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 2, 11, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        const auto third = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 3, 12, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);

        const auto first_admission = owner.accept(first);
        const auto second_admission = owner.accept(second);
        ASSERT_TRUE(first_admission.accepted());
        ASSERT_TRUE(second_admission.accepted());

        const auto blocked = owner.accept(third);
        EXPECT_EQ(
            blocked.status,
            MoEOverlayInferenceAdmissionStatus::Backpressured);
        EXPECT_EQ(owner.nextTransactionOrdinal(), 3u)
            << "Backpressure must not consume the exact next ordinal";

        completeSlot(owner, first_admission, first);
        const auto retried = owner.accept(third);
        ASSERT_TRUE(retried.accepted()) << retried.error;
        EXPECT_EQ(retried.slot_index, 0u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         TerminalWaitsForEveryReturnAndMayBeRetriedExactly)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));
        const auto execute = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 1, 20, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        const auto admitted = owner.accept(execute);
        ASSERT_TRUE(admitted.accepted());

        const auto terminal = makeMoEOverlayInferenceTerminalTicket(
            topology(), active, 2, 41,
            MoEOverlayInferenceTransactionAction::Complete);
        EXPECT_EQ(
            owner.accept(terminal).status,
            MoEOverlayInferenceAdmissionStatus::Backpressured);
        EXPECT_EQ(owner.nextTransactionOrdinal(), 2u);

        completeSlot(owner, admitted, execute);
        EXPECT_TRUE(owner.accept(terminal).completed());
        EXPECT_EQ(
            owner.accept(terminal).status,
            MoEOverlayInferenceAdmissionStatus::Rejected)
            << "A terminal ticket is not replayable";
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         StaleGenerationWorkspaceTopologyAndOrdinalAreRejected)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));
        const auto valid = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 1, 30, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);

        auto stale_request = valid;
        --stale_request.request_generation;
        EXPECT_EQ(owner.accept(stale_request).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);

        auto stale_workspace = valid;
        ++stale_workspace.workspace_generation;
        EXPECT_EQ(owner.accept(stale_workspace).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);

        auto divergent_topology = valid;
        divergent_topology.topology_fingerprint_high ^= 0x10u;
        EXPECT_EQ(owner.accept(divergent_topology).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);

        auto future = valid;
        future.transaction_ordinal = 2;
        EXPECT_EQ(owner.accept(future).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);

        const auto admitted = owner.accept(valid);
        ASSERT_TRUE(admitted.accepted());
        EXPECT_EQ(owner.accept(valid).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         PlacementEpochMayAdvanceBetweenTransactionsButNeverRegress)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));
        const auto first = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 1, 40, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        const auto first_admission = owner.accept(first);
        completeSlot(owner, first_admission, first);

        const auto promoted = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 2, 41, 42,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        const auto promoted_admission = owner.accept(promoted);
        completeSlot(owner, promoted_admission, promoted);
        EXPECT_EQ(owner.currentPlacementEpoch(), 42u);

        const auto regressed = makeMoEOverlayInferenceExecutionTicket(
            topology(), active, 3, 42, 41,
            MoEOverlayInferenceGraphRole::MainDecode, 1, 1, 1);
        EXPECT_EQ(owner.accept(regressed).status,
                  MoEOverlayInferenceAdmissionStatus::Rejected);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         GeometryBuildersRejectInvalidRoleAndDepthCombinations)
    {
        const auto active = command();
        EXPECT_THROW(
            (void)makeMoEOverlayInferenceExecutionTicket(
                topology(), active, 1, 50, 41,
                MoEOverlayInferenceGraphRole::MainDecode, 1, 2, 2),
            std::invalid_argument);
        EXPECT_THROW(
            (void)makeMoEOverlayInferenceExecutionTicket(
                topology(), active, 1, 50, 41,
                MoEOverlayInferenceGraphRole::MTPGroupedVerifier,
                1, 3, 4, 3),
            std::invalid_argument);
        EXPECT_THROW(
            (void)makeMoEOverlayInferenceExecutionTicket(
                topology(), active, 1, 50, 41,
                MoEOverlayInferenceGraphRole::MTPDraft,
                1, 1, 1, 2, 2),
            std::invalid_argument);
        EXPECT_THROW(
            (void)makeMoEOverlayInferenceTerminalTicket(
                topology(), active, 1, 41,
                MoEOverlayInferenceTransactionAction::Complete, 7),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         AbortIsTerminalAndCannotBeHiddenByAReplacementCommand)
    {
        auto owner = protocol();
        const auto active = command();
        ASSERT_TRUE(owner.beginCommand(active));
        const auto abort = makeMoEOverlayInferenceTerminalTicket(
            topology(), active, 1, 41,
            MoEOverlayInferenceTransactionAction::Abort, 17);
        EXPECT_EQ(owner.accept(abort).status,
                  MoEOverlayInferenceAdmissionStatus::Aborted);
        EXPECT_EQ(owner.state(), MoEOverlayInferenceProtocolState::Failed);

        std::string error;
        EXPECT_FALSE(owner.beginCommand(command(10, 1, 42), &error));
        EXPECT_NE(error.find("process-terminal"), std::string::npos);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         CompletedCommandsRequireMonotonicRequestAndCommandIdentity)
    {
        auto owner = protocol();
        const auto first_command = command();
        ASSERT_TRUE(owner.beginCommand(first_command));
        const auto terminal = makeMoEOverlayInferenceTerminalTicket(
            topology(), first_command, 1, 41,
            MoEOverlayInferenceTransactionAction::Complete);
        ASSERT_TRUE(owner.accept(terminal).completed());

        EXPECT_FALSE(owner.beginCommand(first_command));
        EXPECT_TRUE(owner.beginCommand(command(9, 4, 41)));
        const auto second_terminal = makeMoEOverlayInferenceTerminalTicket(
            topology(), command(9, 4, 41), 1, 41,
            MoEOverlayInferenceTransactionAction::Complete);
        ASSERT_TRUE(owner.accept(second_terminal).completed());
        EXPECT_TRUE(owner.beginCommand(command(10, 1, 50)));
    }

} // namespace llaminar2::test
