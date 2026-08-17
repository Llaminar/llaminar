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

#include <gtest/gtest.h>

#include <chrono>
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
                        descriptor.sidecar_depth);
                }
                catch (const std::exception &exception)
                {
                    result.error = exception.what();
                    return result;
                }
                result.ok = true;
                result.slot_index = publish_count_++;
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

        private:
            MoEOverlayInferenceTopologyIdentity topology_;
            MoEOverlayInferenceCommandIdentity command_{};
            bool active_ = false;
            std::uint64_t next_ordinal_ = 1;
            std::size_t publish_count_ = 0;
            std::size_t retire_count_ = 0;
            std::size_t complete_count_ = 0;
            std::size_t abort_count_ = 0;
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

        MoEExpertOwnerMap heterogeneousOwnerMap()
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
            return MoEExpertOwnerMap::build(plan);
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
         RankCoordinatorPublishesOnceAcrossSymmetricParticipantsAndTargets)
    {
        auto first_target = std::make_shared<RecordingPublisher>(1);
        auto second_target = std::make_shared<RecordingPublisher>(2);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {first_target, second_target},
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
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

    TEST(Test__MoEOverlayInferenceTransaction,
         RankCoordinatorRejectsDivergentParticipantGeometry)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
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
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 1,
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
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
                    .max_transactions_per_command = 4,
                    .max_mtp_draft_depth = 3,
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

        submit_chunk();
        EXPECT_EQ(publisher->publishCount(), 2u);
        EXPECT_EQ(publisher->retireCount(), 1u)
            << "The first complete chunk must release its fixed protocol slot "
               "before the next chunk publishes";

        ASSERT_TRUE(coordinator->completeCommand(41));
        EXPECT_EQ(publisher->retireCount(), 2u);
        EXPECT_EQ(publisher->completeCount(), 1u);
    }

    TEST(Test__MoEOverlayInferenceTransaction,
         AheadPrefillParticipantWaitsForSiblingBeforeEnteringNextChunk)
    {
        auto publisher = std::make_shared<RecordingPublisher>(1);
        auto coordinator =
            std::make_shared<MoEOverlayInferenceTransactionCoordinator>(
                MoEOverlayInferenceTransactionCoordinator::Config{
                    .publishers = {publisher},
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
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
                    .continuation_participant_count = 2,
                    .ticket_authority_participant_index = 0,
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
