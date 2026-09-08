/**
 * @file Test__DeviceGenerationController.cpp
 * @brief Exhaustive host proof for the device-owned MTP response controller.
 *
 * GPU generation deliberately leaves response assembly, speculative carry
 * state, request budgeting, and terminal status on device between verifier
 * transactions.  CUDA and ROCm invoke the host/device helpers tested here from
 * graph-captured kernels.  These tests therefore define the serial-decode
 * contract independently of either GPU compiler and make malformed lifecycle
 * transitions fail visibly rather than permitting a host-side repair.
 */

#include <gtest/gtest.h>

#include "backends/IGPUGraphCapture.h"
#include "execution/mtp/HostedDeviceGenerationLifecycle.h"
#include "kernels/common/SamplingMath.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace
{
    using llaminar2::DeviceControlledLoopFragment;
    using llaminar2::DeviceControlledLoopFragmentExecution;
    using llaminar2::DeviceControlledLoopTicketSelection;
    using namespace llaminar2::sampling_math;

    using ControlRow = std::array<int, kDeviceGenerationControlCount>;
    using MetaRow = std::array<int, kSpeculativeBatchMetaCount>;

    /**
     * @brief Prove the controller handoff admits only the complete transaction cycle.
     */
    TEST(
        Test__DeviceGenerationController,
        StateHandoffRequiresPublishBorrowRepublishOrder)
    {
        using llaminar2::DeviceGenerationStateHandoff;
        using llaminar2::DeviceGenerationStatePublicationKind;
        using llaminar2::DeviceTimelineRole;

        DeviceGenerationStateHandoff handoff;
        void *const admission_stream = reinterpret_cast<void *>(0x1000);
        void *const verifier_stream = reinterpret_cast<void *>(0x2000);
        void *const terminal_stream = reinterpret_cast<void *>(0x3000);

        EXPECT_TRUE(handoff.inactive());
        ASSERT_TRUE(handoff.publish(
            DeviceGenerationStatePublicationKind::Admission,
            admission_stream,
            /*request_count=*/2,
            [] { return true; }));
        EXPECT_TRUE(handoff.published());
        EXPECT_EQ(handoff.frontierStream(), admission_stream);
        EXPECT_EQ(handoff.requestCount(), 2);

        EXPECT_FALSE(handoff.publish(
            DeviceGenerationStatePublicationKind::CommittedTransaction,
            admission_stream,
            2,
            [] { return true; }))
            << "A published version must be borrowed before it can be replaced.";

        ASSERT_TRUE(handoff.borrow(
            verifier_stream,
            DeviceTimelineRole::AllPositionVerifier,
            2,
            [&](void *producer, void *consumer)
            {
                EXPECT_EQ(producer, admission_stream);
                EXPECT_EQ(consumer, verifier_stream);
                return true;
            }));
        EXPECT_TRUE(handoff.borrowed());
        EXPECT_EQ(
            handoff.borrowerRole(),
            DeviceTimelineRole::AllPositionVerifier);

        ASSERT_TRUE(handoff.publish(
            DeviceGenerationStatePublicationKind::Terminal,
            terminal_stream,
            2,
            [] { return true; }));
        EXPECT_TRUE(handoff.published());
        EXPECT_EQ(
            handoff.lastPublication(),
            DeviceGenerationStatePublicationKind::Terminal);

        ASSERT_TRUE(handoff.borrow(
            terminal_stream,
            DeviceTimelineRole::HostResultBridge,
            2,
            [](void *, void *) { return true; }));
        ASSERT_TRUE(handoff.retireAfterHostCompletion(2));
        EXPECT_TRUE(handoff.inactive());
    }

    /**
     * @brief Failed backend submissions must leave lifecycle authority unchanged.
     */
    TEST(
        Test__DeviceGenerationController,
        StateHandoffCommitsOnlyAfterBackendEdgeSucceeds)
    {
        using llaminar2::DeviceGenerationStateHandoff;
        using llaminar2::DeviceGenerationStatePublicationKind;
        using llaminar2::DeviceTimelineRole;

        DeviceGenerationStateHandoff handoff;
        void *const producer = reinterpret_cast<void *>(0x4000);
        void *const consumer = reinterpret_cast<void *>(0x5000);

        EXPECT_FALSE(handoff.publish(
            DeviceGenerationStatePublicationKind::Admission,
            producer,
            1,
            [] { return false; }));
        EXPECT_TRUE(handoff.inactive());

        ASSERT_TRUE(handoff.publish(
            DeviceGenerationStatePublicationKind::Admission,
            producer,
            1,
            [] { return true; }));
        EXPECT_FALSE(handoff.borrow(
            consumer,
            DeviceTimelineRole::AllPositionVerifier,
            1,
            [](void *, void *) { return false; }));
        EXPECT_TRUE(handoff.published());
        EXPECT_EQ(handoff.frontierStream(), producer);
    }

    /**
     * @brief Reset can retire both an untouched publication and a failed borrow.
     */
    TEST(
        Test__DeviceGenerationController,
        StateHandoffResetRetiresPublishedAndBorrowedFrontiers)
    {
        using llaminar2::DeviceGenerationStateHandoff;
        using llaminar2::DeviceGenerationStateHandoffPhase;
        using llaminar2::DeviceGenerationStatePublicationKind;
        using llaminar2::DeviceTimelineRole;

        void *const producer = reinterpret_cast<void *>(0x6000);
        void *const borrower = reinterpret_cast<void *>(0x7000);
        void *const reset = reinterpret_cast<void *>(0x8000);

        for (const bool borrow_before_reset : {false, true})
        {
            DeviceGenerationStateHandoff handoff;
            ASSERT_TRUE(handoff.publish(
                DeviceGenerationStatePublicationKind::Admission,
                producer,
                1,
                [] { return true; }));
            if (borrow_before_reset)
            {
                ASSERT_TRUE(handoff.borrow(
                    borrower,
                    DeviceTimelineRole::AllPositionVerifier,
                    1,
                    [](void *, void *) { return true; }));
            }

            ASSERT_TRUE(handoff.retireForReset(
                reset,
                1,
                [&](DeviceGenerationStateHandoffPhase phase,
                    void *frontier,
                    DeviceTimelineRole role,
                    void *reset_stream)
                {
                    EXPECT_EQ(reset_stream, reset);
                    EXPECT_EQ(
                        phase,
                        borrow_before_reset
                            ? DeviceGenerationStateHandoffPhase::Borrowed
                            : DeviceGenerationStateHandoffPhase::Published);
                    EXPECT_EQ(
                        frontier,
                        borrow_before_reset ? borrower : producer);
                    EXPECT_EQ(
                        role,
                        borrow_before_reset
                            ? DeviceTimelineRole::AllPositionVerifier
                            : DeviceTimelineRole::Count);
                    return true;
                }));
            EXPECT_TRUE(handoff.inactive());
        }
    }

    TEST(
        Test__DeviceGenerationController,
        HostedTicketSelectionPreservesDepthPrefixThenMaintenanceOrder)
    {
        const std::array<DeviceControlledLoopFragment, 5> branch = {{
            {.name = "transaction",
             .execution = DeviceControlledLoopFragmentExecution::Always},
            {.name = "depth two",
             .execution = DeviceControlledLoopFragmentExecution::
                 IfDeviceSelectorAtLeast,
             .minimum_selector = 2},
            {.name = "depth three",
             .execution = DeviceControlledLoopFragmentExecution::
                 IfDeviceSelectorAtLeast,
             .minimum_selector = 3},
            {.name = "epoch release",
             .execution = DeviceControlledLoopFragmentExecution::Always},
            {.name = "maintenance",
             .execution = DeviceControlledLoopFragmentExecution::
                 IfDeviceWordNonZero},
        }};
        const std::span<const DeviceControlledLoopFragment> ordered_branch(
            branch);

        const DeviceControlledLoopTicketSelection due{
            .iteration_admitted = true,
            .conditional_word_nonzero = true,
            .selector = 3,
        };
        ASSERT_EQ(due.countSelected(ordered_branch), 5u);
        ASSERT_NE(due.selectOrdinal(ordered_branch, 0u), nullptr);
        ASSERT_NE(due.selectOrdinal(ordered_branch, 1u), nullptr);
        ASSERT_NE(due.selectOrdinal(ordered_branch, 2u), nullptr);
        ASSERT_NE(due.selectOrdinal(ordered_branch, 3u), nullptr);
        ASSERT_NE(due.selectOrdinal(ordered_branch, 4u), nullptr);
        EXPECT_STREQ(due.selectOrdinal(ordered_branch, 0u)->name,
                     "transaction");
        EXPECT_STREQ(due.selectOrdinal(ordered_branch, 1u)->name,
                     "depth two");
        EXPECT_STREQ(due.selectOrdinal(ordered_branch, 2u)->name,
                     "depth three");
        EXPECT_STREQ(due.selectOrdinal(ordered_branch, 3u)->name,
                     "epoch release");
        EXPECT_STREQ(due.selectOrdinal(ordered_branch, 4u)->name,
                     "maintenance");

        const DeviceControlledLoopTicketSelection not_due{
            .iteration_admitted = true,
            .conditional_word_nonzero = false,
            .selector = 2,
        };
        EXPECT_EQ(not_due.countSelected(ordered_branch), 3u);
        ASSERT_NE(not_due.selectOrdinal(ordered_branch, 2u), nullptr);
        EXPECT_STREQ(
            not_due.selectOrdinal(ordered_branch, 2u)->name,
            "epoch release");
        EXPECT_EQ(not_due.selectOrdinal(ordered_branch, 3u), nullptr);

        const DeviceControlledLoopTicketSelection terminal{
            .iteration_admitted = false,
            .conditional_word_nonzero = true,
            .selector = 3,
        };
        EXPECT_EQ(terminal.countSelected(ordered_branch), 0u);
        EXPECT_EQ(terminal.selectOrdinal(ordered_branch, 0u), nullptr);
    }

    TEST(
        Test__DeviceGenerationController,
        WordPredicatePolarityIsTypedAndPreservedByHostedTickets)
    {
        uint32_t word = 0;
        const auto *capture_identity =
            reinterpret_cast<const llaminar2::IGPUGraphCapture *>(
                static_cast<std::uintptr_t>(1));
        DeviceControlledLoopFragment pending{
            .name = "pending prefill sample",
            .capture = capture_identity,
            .execution = DeviceControlledLoopFragmentExecution::IfDeviceWordZero,
            .condition_word_device = &word};
        ASSERT_TRUE(pending.valid());
        auto consumed = pending;
        consumed.execution = DeviceControlledLoopFragmentExecution::IfDeviceWordNonZero;
        ASSERT_TRUE(consumed.valid());
        EXPECT_FALSE(pending.hasSameExecutionIdentity(consumed));
        auto incomplete = pending;
        incomplete.condition_word_device = nullptr;
        EXPECT_FALSE(incomplete.valid());
        auto ambiguous = pending;
        ambiguous.minimum_selector = 0;
        EXPECT_FALSE(ambiguous.valid());

        for (bool admitted : {false, true})
        for (bool nonzero : {false, true})
        {
            const DeviceControlledLoopTicketSelection ticket{
                .iteration_admitted = admitted,
                .conditional_word_nonzero = nonzero};
            EXPECT_EQ(ticket.selects(pending), admitted && !nonzero);
            EXPECT_EQ(ticket.selects(consumed), admitted && nonzero);
        }
    }

    TEST(
        Test__DeviceGenerationController,
        SelectorFragmentPolicyRejectsAmbiguousBindingsAndKeysThreshold)
    {
        const auto *capture_identity =
            reinterpret_cast<const llaminar2::IGPUGraphCapture *>(
                static_cast<std::uintptr_t>(1));
        const DeviceControlledLoopFragment selector_fragment{
            .name = "depth prefix",
            .capture = capture_identity,
            .execution = DeviceControlledLoopFragmentExecution::
                IfDeviceSelectorAtLeast,
            .minimum_selector = 7,
        };
        ASSERT_TRUE(selector_fragment.valid());

        DeviceControlledLoopFragment different_threshold =
            selector_fragment;
        different_threshold.minimum_selector = 8;
        ASSERT_TRUE(different_threshold.valid());
        EXPECT_FALSE(selector_fragment.hasSameExecutionIdentity(
            different_threshold));

        DeviceControlledLoopFragment ambiguous = selector_fragment;
        ambiguous.condition_word_device =
            reinterpret_cast<const uint32_t *>(
                static_cast<std::uintptr_t>(4));
        EXPECT_FALSE(ambiguous.valid());

        DeviceControlledLoopFragment missing_threshold =
            selector_fragment;
        missing_threshold.minimum_selector = -1;
        EXPECT_FALSE(missing_threshold.valid());
    }

    /**
     * @brief Construct one valid compact verifier metadata row.
     */
    MetaRow makeMeta(
        int output_count,
        int leading_count,
        int verifier_state_count,
        int accepted_prefix,
        int consumed_rows,
        bool all_accepted,
        bool stopped = false,
        bool commit_boundary_clipped = false)
    {
        MetaRow meta{};
        meta[kSpecBatchMetaOk] = 1;
        meta[kSpecBatchMetaOutputCount] = output_count;
        meta[kSpecBatchMetaAcceptedSpeculativePrefix] = accepted_prefix;
        meta[kSpecBatchMetaTargetVerifierStateCommitCount] =
            verifier_state_count;
        meta[kSpecBatchMetaStoppedOnOutput] = stopped ? 1 : 0;
        meta[kSpecBatchMetaAllSpeculativeAccepted] = all_accepted ? 1 : 0;
        meta[kSpecBatchMetaConsumedVerifierRows] = consumed_rows;
        meta[kSpecBatchMetaSampledTerminal] = all_accepted ? 1 : 0;
        meta[kSpecBatchMetaCommitBoundaryClipped] =
            commit_boundary_clipped ? 1 : 0;
        meta[kSpecBatchMetaLeadingCommittedOutputCount] = leading_count;
        return meta;
    }

    /**
     * @brief Assert the controller's fail-hard terminal state.
     */
    void expectFatal(
        const ControlRow &control,
        DeviceGenerationError expected_error)
    {
        EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
        EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
        EXPECT_EQ(control[kDeviceGenerationControlTransactionCommitBudget], 0);
        EXPECT_EQ(
            control[kDeviceGenerationControlErrorCode],
            static_cast<int>(expected_error));
    }

    /**
     * @brief Construct an explicit fixed-depth admission policy for one test.
     */
    DeviceGenerationPolicy fixedDepthPolicy(int depth)
    {
        return DeviceGenerationPolicy::fixed(depth);
    }

    /**
     * @brief Prove that durable verifier identity uses the committed depth.
     *
     * Dynamic policy observation may choose the next depth as part of the same
     * controller commit.  The diagnostic record must retain the depth and
     * tokens of the transaction just retired, not reinterpret the reusable row
     * with that next policy decision.
     */
    TEST(
        Test__DeviceGenerationController,
        CommittedVerifierIdentityIsCompleteAndUsesLastTransactionDepth)
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            16, 16, fixedDepthPolicy(3), control.data()));

        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> verifier{};
        verifier.fill(-1);
        verifier[0] = 101;
        verifier[1] = 201;
        verifier[2] = 202;
        verifier[3] = 203;
        ASSERT_TRUE(valid_committed_verifier_identity_input(
            verifier.data(), verifier.size(), control.data()));

        control[kDeviceGenerationControlTransactionCount] = 4;
        control[kDeviceGenerationControlLastTransactionDraftDepth] = 3;
        /* Model the adaptive controller selecting depth one for the next turn. */
        control[kDeviceGenerationControlCurrentDraftDepth] = 1;

        MTPCommittedVerifierIdentityRecord identity{};
        ASSERT_TRUE(publish_committed_verifier_identity(
            verifier.data(), verifier.size(), control.data(), &identity));
        EXPECT_EQ(identity.valid, 1u);
        EXPECT_EQ(identity.version, kMTPCommittedVerifierIdentityVersion);
        EXPECT_EQ(identity.transaction_count, 4);
        EXPECT_EQ(identity.draft_depth, 3);
        EXPECT_EQ(identity.verifier_input_tokens[0], 101);
        EXPECT_EQ(identity.verifier_input_tokens[1], 201);
        EXPECT_EQ(identity.verifier_input_tokens[2], 202);
        EXPECT_EQ(identity.verifier_input_tokens[3], 203);
        for (int row = 4; row < kSpeculativeBatchMaxOutputTokens; ++row)
            EXPECT_EQ(identity.verifier_input_tokens[row], -1);
    }

    TEST(
        Test__DeviceGenerationController,
        CommittedVerifierIdentityRejectsSentinelAndShortActiveRows)
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            16, 16, fixedDepthPolicy(3), control.data()));
        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> verifier{};
        verifier.fill(-1);
        verifier[0] = 101;
        verifier[1] = 201;
        verifier[2] = 202;

        EXPECT_FALSE(valid_committed_verifier_identity_input(
            verifier.data(), verifier.size(), control.data()));
        verifier[3] = 203;
        EXPECT_FALSE(valid_committed_verifier_identity_input(
            verifier.data(), 3, control.data()));
        EXPECT_TRUE(valid_committed_verifier_identity_input(
            verifier.data(), verifier.size(), control.data()));
    }
} // namespace

TEST(Test__DeviceGenerationController, InitializationAndBudgetAreTotalForPositiveSizes)
{
    using namespace llaminar2::sampling_math;

    const std::array<int, 12> response_budgets = {
        1, 2, 3, 4, 8, 15, 16, 31, 64, 255, 1024, 4096};
    for (const int response_budget : response_budgets)
    {
        for (int draft_depth = 1; draft_depth <= 15; ++draft_depth)
        {
            const int verifier_rows = draft_depth + 1;
            ControlRow control;
            control.fill(-1);
            ASSERT_TRUE(initialize_device_generation_control(
                response_budget,
                response_budget + 7,
                fixedDepthPolicy(draft_depth),
                control.data()));

            EXPECT_EQ(control[kDeviceGenerationControlOk], 1);
            EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 0);
            EXPECT_EQ(
                control[kDeviceGenerationControlRemainingTokenCount],
                response_budget);
            EXPECT_EQ(
                control[kDeviceGenerationControlCurrentDraftDepth],
                draft_depth);
            EXPECT_EQ(
                control[kDeviceGenerationControlActiveVerifierRowCount],
                verifier_rows);
            EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 0);
            EXPECT_EQ(
                control[kDeviceGenerationControlPublishedStateCommitCount],
                0);
            EXPECT_EQ(
                control[kDeviceGenerationControlErrorCode],
                static_cast<int>(DeviceGenerationError::None));
            EXPECT_EQ(
                control[
                    kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount],
                0);
            EXPECT_EQ(
                control[
                    kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount],
                0);

            for (int maintenance_rows = 1; maintenance_rows <= 16;
                 ++maintenance_rows)
            {
                ControlRow candidate = control;
                const int actual =
                    prepare_device_generation_transaction_budget(
                        verifier_rows,
                        maintenance_rows,
                        candidate.data());
                EXPECT_EQ(
                    actual,
                    std::min({
                        response_budget,
                        verifier_rows,
                        maintenance_rows}));
                EXPECT_EQ(
                    candidate[kDeviceGenerationControlTransactionCommitBudget],
                    actual);
            }
        }
    }
}

TEST(Test__DeviceGenerationController,
     DispatchTicketExposesOnlyAuthenticatedSchedulingState)
{
    constexpr uint64_t session_epoch = 0x0123456789ABCDEFull;
    constexpr uint64_t workspace_generation = 0xFEDCBA9876543210ull;

    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/32,
        /*response_capacity=*/64,
        fixedDepthPolicy(/*depth=*/4),
        control.data()));

    DeviceGenerationDispatchTicket ticket{};
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(
        session_epoch,
        workspace_generation,
        &ticket));
    EXPECT_TRUE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation));
    EXPECT_EQ(ticket.transaction_count, 0);
    EXPECT_EQ(ticket.committed_output_tokens, 0);
    EXPECT_EQ(ticket.healthy, 0)
        << "Admission identity alone must not advertise a schedulable row.";

    control[kDeviceGenerationControlTransactionCount] = 7;
    control[kDeviceGenerationControlResponseTokenCount] = 19;
    control[kDeviceGenerationControlCurrentDraftDepth] = 4;
    control[kDeviceGenerationControlRequestComplete] = 0;
    control[kDeviceGenerationControlErrorCode] =
        static_cast<int>(DeviceGenerationError::None);
    const uint32_t maintenance_due = 1;
    ASSERT_TRUE(publish_device_generation_dispatch_ticket(
        control.data(),
        &maintenance_due,
        &ticket));

    EXPECT_TRUE(ticket.matchesLifecycle(
        session_epoch,
        workspace_generation));
    EXPECT_EQ(ticket.healthy, 1);
    EXPECT_EQ(ticket.complete, 0);
    EXPECT_EQ(ticket.transaction_count, 7);
    EXPECT_EQ(ticket.next_draft_depth, 4);
    EXPECT_EQ(ticket.error_code, static_cast<int>(DeviceGenerationError::None));
    EXPECT_EQ(ticket.maintenance_due, 1);
    EXPECT_EQ(ticket.committed_output_tokens, 19);

    DeviceGenerationDispatchTicket mirrored_ticket = ticket;
    mirrored_ticket.session_epoch_low ^= 0x1u;
    EXPECT_FALSE(mirrored_ticket.matchesLifecycle(
        session_epoch,
        workspace_generation));
    EXPECT_TRUE(ticket.hasSameDispatchDecision(mirrored_ticket))
        << "Participant-local lifecycle identity is not a dispatch decision.";
    mirrored_ticket.next_draft_depth = 3;
    EXPECT_FALSE(ticket.hasSameDispatchDecision(mirrored_ticket));
    mirrored_ticket = ticket;
    ++mirrored_ticket.committed_output_tokens;
    EXPECT_FALSE(ticket.hasSameDispatchDecision(mirrored_ticket));
}

TEST(Test__DeviceGenerationController,
     DispatchTicketPoisonsMalformedMaintenanceState)
{
    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/8,
        /*response_capacity=*/8,
        fixedDepthPolicy(/*depth=*/2),
        control.data()));
    control[kDeviceGenerationControlTransactionCount] = 1;

    DeviceGenerationDispatchTicket ticket{};
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(
        /*session_epoch=*/11,
        /*workspace_generation=*/29,
        &ticket));
    const uint32_t poisoned_maintenance_due = 2;
    ASSERT_TRUE(publish_device_generation_dispatch_ticket(
        control.data(),
        &poisoned_maintenance_due,
        &ticket));

    expectFatal(control, DeviceGenerationError::InvalidMaintenanceState);
    EXPECT_EQ(ticket.healthy, 0);
    EXPECT_EQ(ticket.complete, 1);
    EXPECT_EQ(
        ticket.error_code,
        static_cast<int>(DeviceGenerationError::InvalidMaintenanceState));
    EXPECT_EQ(ticket.maintenance_due, 0)
        << "A poisoned maintenance value must never select a graph branch.";
}

TEST(Test__DeviceGenerationController,
     DynamicDispatchTicketPublishesEverySupportedDepthWithoutHostPolicy)
{
    DeviceGenerationPolicy policy;
    policy.mode = DeviceGenerationPolicyMode::Dynamic;
    policy.initial_depth = 1;
    policy.minimum_depth = 1;
    policy.maximum_depth =
        DeviceGenerationPolicy::kMaximumSupportedDraftDepth;
    ASSERT_TRUE(policy.valid());

    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/256,
        /*response_capacity=*/256,
        policy,
        control.data()));

    DeviceGenerationDispatchTicket ticket{};
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(
        /*session_epoch=*/77,
        /*workspace_generation=*/91,
        &ticket));

    for (int depth = policy.minimum_depth;
         depth <= policy.maximum_depth;
         ++depth)
    {
        control[kDeviceGenerationControlTransactionCount] = depth;
        control[kDeviceGenerationControlResponseTokenCount] = depth + 7;
        control[kDeviceGenerationControlCurrentDraftDepth] = depth;
        control[kDeviceGenerationControlActiveVerifierRowCount] = depth + 1;
        const uint32_t maintenance_due =
            static_cast<uint32_t>(depth & 1);
        ASSERT_TRUE(publish_device_generation_dispatch_ticket(
            control.data(),
            &maintenance_due,
            &ticket));
        EXPECT_TRUE(ticket.matchesLifecycle(77, 91));
        EXPECT_EQ(ticket.transaction_count, depth);
        EXPECT_EQ(ticket.next_draft_depth, depth);
        EXPECT_EQ(ticket.committed_output_tokens, depth + 7);
        EXPECT_EQ(
            ticket.maintenance_due,
            static_cast<int32_t>(maintenance_due));
        EXPECT_EQ(ticket.healthy, 1);
        EXPECT_EQ(ticket.complete, 0);
    }
}

TEST(Test__DeviceGenerationController,
     CurrentBatchLLEPEvidenceFieldsAreContiguousTerminalControlWords)
{
    using namespace llaminar2::sampling_math;

    EXPECT_EQ(
        kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount + 1,
        kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount);
    EXPECT_EQ(
        kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount +
            1,
        kDeviceGenerationControlCount);
}

TEST(Test__DeviceGenerationController, DynamicPolicyPromotesAndDemotesAcrossEveryDepth)
{
    using namespace llaminar2::sampling_math;

    DeviceGenerationPolicy promote_policy;
    promote_policy.mode = DeviceGenerationPolicyMode::Dynamic;
    promote_policy.initial_depth = 1;
    promote_policy.minimum_depth = 1;
    promote_policy.maximum_depth = 15;
    promote_policy.window_size = 1;
    promote_policy.minimum_samples = 1;
    promote_policy.cooldown_steps = 0;
    promote_policy.promote_consecutive_windows = 1;
    ASSERT_TRUE(promote_policy.valid());

    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/4096,
        /*response_capacity=*/4096,
        promote_policy,
        control.data()));
    for (int depth = 1; depth < 15; ++depth)
    {
        ASSERT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], depth);
        ASSERT_TRUE(record_device_generation_depth_observation(
            /*accepted_prefix=*/depth,
            /*rollback=*/false,
            /*budget_limited=*/false,
            control.data()));
        EXPECT_EQ(
            control[kDeviceGenerationControlCurrentDraftDepth],
            depth + 1);
        EXPECT_EQ(
            control[kDeviceGenerationControlActiveVerifierRowCount],
            depth + 2);
    }
    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotions], 14);
    EXPECT_EQ(control[kDeviceGenerationControlDepthUpdates], 14);

    DeviceGenerationPolicy demote_policy = promote_policy;
    demote_policy.initial_depth = 15;
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/4096,
        /*response_capacity=*/4096,
        demote_policy,
        control.data()));
    for (int depth = 15; depth > 1; --depth)
    {
        ASSERT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], depth);
        ASSERT_TRUE(record_device_generation_depth_observation(
            /*accepted_prefix=*/0,
            /*rollback=*/true,
            /*budget_limited=*/false,
            control.data()));
        EXPECT_EQ(
            control[kDeviceGenerationControlCurrentDraftDepth],
            depth - 1);
        EXPECT_EQ(
            control[kDeviceGenerationControlActiveVerifierRowCount],
            depth);
    }
    EXPECT_EQ(control[kDeviceGenerationControlDepthDemotions], 14);
    EXPECT_EQ(control[kDeviceGenerationControlDepthUpdates], 14);
}

TEST(Test__DeviceGenerationController, ObservePolicyCannotMutateActiveGeometry)
{
    using namespace llaminar2::sampling_math;

    DeviceGenerationPolicy policy;
    policy.mode = DeviceGenerationPolicyMode::Observe;
    policy.initial_depth = 4;
    policy.minimum_depth = 1;
    policy.maximum_depth = 15;
    policy.window_size = 1;
    policy.minimum_samples = 1;
    policy.cooldown_steps = 0;
    policy.promote_consecutive_windows = 1;
    ASSERT_TRUE(policy.valid());

    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/64,
        /*response_capacity=*/64,
        policy,
        control.data()));
    ASSERT_TRUE(record_device_generation_depth_observation(
        /*accepted_prefix=*/4,
        /*rollback=*/false,
        /*budget_limited=*/false,
        control.data()));

    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 4);
    EXPECT_EQ(control[kDeviceGenerationControlActiveVerifierRowCount], 5);
    EXPECT_EQ(control[kDeviceGenerationControlDepthLastRecommendedDepth], 5);
    EXPECT_EQ(control[kDeviceGenerationControlDepthUpdates], 0);
    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotions], 0);
    EXPECT_EQ(control[kDeviceGenerationControlDepthEvaluatedWindows], 1);
}

TEST(Test__DeviceGenerationController,
     DynamicOutcomeLedgerAccumulatesEverySelectedWidthExactly)
{
    using namespace llaminar2::sampling_math;

    DeviceGenerationPolicy policy;
    policy.mode = DeviceGenerationPolicyMode::Dynamic;
    policy.initial_depth = 2;
    policy.minimum_depth = 1;
    policy.maximum_depth = 4;
    policy.window_size = 1;
    policy.minimum_samples = 1;
    policy.cooldown_steps = 0;
    policy.promote_consecutive_windows = 1;
    ASSERT_TRUE(policy.valid());

    ControlRow control{};
    std::array<int32_t, 16> response{};
    response.fill(-1);
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/16,
        static_cast<int>(response.size()),
        policy,
        control.data()));

    const std::array<int32_t, 3> depth_two_tokens = {10, 11, 12};
    const MetaRow depth_two_meta = makeMeta(
        /*output_count=*/3,
        /*leading_count=*/0,
        /*verifier_state_count=*/3,
        /*accepted_prefix=*/2,
        /*consumed_rows=*/2,
        /*all_accepted=*/true);
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            /*verifier_row_capacity=*/5,
            /*maintenance_rows_remaining=*/16,
            control.data()),
        3);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        depth_two_tokens.data(),
        static_cast<int>(depth_two_tokens.size()),
        depth_two_meta.data(),
        static_cast<int>(depth_two_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));
    ASSERT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 3);

    const std::array<int32_t, 4> depth_three_tokens = {20, 21, 22, 23};
    const MetaRow depth_three_meta = makeMeta(
        /*output_count=*/4,
        /*leading_count=*/0,
        /*verifier_state_count=*/4,
        /*accepted_prefix=*/3,
        /*consumed_rows=*/3,
        /*all_accepted=*/true);
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            /*verifier_row_capacity=*/5,
            /*maintenance_rows_remaining=*/16,
            control.data()),
        4);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        depth_three_tokens.data(),
        static_cast<int>(depth_three_tokens.size()),
        depth_three_meta.data(),
        static_cast<int>(depth_three_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));

    EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], 2);
    EXPECT_EQ(control[kDeviceGenerationControlAttemptedDraftTokenCount], 5);
    EXPECT_EQ(control[kDeviceGenerationControlVerifierTokenCount], 7);
    EXPECT_EQ(control[kDeviceGenerationControlLastTransactionDraftDepth], 3);
    EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 4);
    EXPECT_EQ(control[kDeviceGenerationControlCurrentDraftDepth], 4);
    EXPECT_EQ(control[kDeviceGenerationControlDepthUpdates], 2);
    EXPECT_EQ(control[kDeviceGenerationControlDepthPromotions], 2);
    EXPECT_EQ(control[kDeviceGenerationControlDepthDemotions], 0);
}

TEST(Test__DeviceGenerationController, InvalidPolicyAndSelectorFailHard)
{
    using namespace llaminar2::sampling_math;

    ControlRow control{};
    const DeviceGenerationPolicy invalid_policy =
        DeviceGenerationPolicy::fixed(0);
    EXPECT_FALSE(initialize_device_generation_control(
        /*max_new_tokens=*/8,
        /*response_capacity=*/8,
        invalid_policy,
        control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
    EXPECT_EQ(
        control[kDeviceGenerationControlErrorCode],
        static_cast<int>(DeviceGenerationError::InvalidDepthPolicy));

    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/8,
        /*response_capacity=*/8,
        fixedDepthPolicy(4),
        control.data()));
    EXPECT_EQ(
        prepare_device_generation_transaction_budget(
            /*verifier_row_capacity=*/4,
            /*maintenance_rows_remaining=*/4,
            control.data()),
        0);
    expectFatal(control, DeviceGenerationError::InvalidDepthSelector);
}

TEST(Test__DeviceGenerationController, PublicationRejectsBudgetBeyondActiveCapturedRows)
{
    using namespace llaminar2::sampling_math;

    constexpr int active_verifier_rows = 5;
    constexpr int configured_max_verifier_rows = 15;
    ControlRow control{};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/64,
        /*response_capacity=*/64,
        fixedDepthPolicy(active_verifier_rows - 1),
        control.data()));
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            configured_max_verifier_rows,
            configured_max_verifier_rows,
            control.data()),
        active_verifier_rows);
    control[kDeviceGenerationControlTransactionCommitBudget] =
        configured_max_verifier_rows;

    std::array<int32_t, active_verifier_rows> compact_tokens = {
        101, 102, 103, -1, -1};
    MetaRow meta = makeMeta(
        /*output_count=*/3,
        /*leading_count=*/0,
        /*verifier_state_count=*/2,
        /*accepted_prefix=*/1,
        /*consumed_rows=*/2,
        /*all_accepted=*/false);
    std::array<int32_t, 64> response{};
    int restore_row = -1;
    int target_cached_tokens = -1;
    int accepted_state_count = -1;
    int publication_ok = -1;
    int32_t next_condition_token = -1;

    EXPECT_FALSE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            compact_tokens.data(),
            static_cast<int>(compact_tokens.size()),
            meta.data(),
            static_cast<int>(meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/active_verifier_rows,
            /*base_cached_tokens=*/23,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token));

    expectFatal(control, DeviceGenerationError::InvalidPublicationMetadata);
    EXPECT_EQ(meta[kSpecBatchMetaOk], 0);
    EXPECT_EQ(publication_ok, 0);
}

TEST(Test__DeviceGenerationController, RejectCarryAndStopMatchSerialResponseBytes)
{
    using namespace llaminar2::sampling_math;

    ControlRow control{};
    std::array<int32_t, 16> response{};
    response.fill(-1);
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/8,
        static_cast<int>(response.size()),
        fixedDepthPolicy(3),
        control.data()));

    const std::array<int32_t, 4> first_tokens = {10, 11, 12, 13};
    const MetaRow first_meta = makeMeta(
        /*output_count=*/4,
        /*leading_count=*/0,
        /*verifier_state_count=*/3,
        /*accepted_prefix=*/3,
        /*consumed_rows=*/3,
        /*all_accepted=*/true);
    ASSERT_EQ(prepare_device_generation_transaction_budget(4, 4, control.data()), 4);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        first_tokens.data(),
        static_cast<int>(first_tokens.size()),
        first_meta.data(),
        static_cast<int>(first_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));

    const std::array<int32_t, 3> second_tokens = {13, 20, 21};
    const MetaRow second_meta = makeMeta(
        /*output_count=*/3,
        /*leading_count=*/1,
        /*verifier_state_count=*/2,
        /*accepted_prefix=*/1,
        /*consumed_rows=*/2,
        /*all_accepted=*/false);
    ASSERT_EQ(prepare_device_generation_transaction_budget(4, 4, control.data()), 4);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        second_tokens.data(),
        static_cast<int>(second_tokens.size()),
        second_meta.data(),
        static_cast<int>(second_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));

    const std::array<int32_t, 2> stop_tokens = {21, 2};
    const MetaRow stop_meta = makeMeta(
        /*output_count=*/2,
        /*leading_count=*/1,
        /*verifier_state_count=*/1,
        /*accepted_prefix=*/0,
        /*consumed_rows=*/1,
        /*all_accepted=*/false,
        /*stopped=*/true);
    ASSERT_EQ(prepare_device_generation_transaction_budget(4, 4, control.data()), 2);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        stop_tokens.data(),
        static_cast<int>(stop_tokens.size()),
        stop_meta.data(),
        static_cast<int>(stop_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));

    const std::array<int32_t, 7> expected = {10, 11, 12, 13, 20, 21, 2};
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), response.begin()));
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 7);
    EXPECT_EQ(control[kDeviceGenerationControlRemainingTokenCount], 1);
    EXPECT_EQ(control[kDeviceGenerationControlModelStopped], 1);
    EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
    EXPECT_EQ(control[kDeviceGenerationControlTransactionCount], 3);
    EXPECT_EQ(control[kDeviceGenerationControlNextLeadingCommittedOutputCount], 0);
    EXPECT_EQ(control[kDeviceGenerationControlAcceptedSpeculativeTokenCount], 4);
    EXPECT_EQ(control[kDeviceGenerationControlRejectedTransactionCount], 1);
    EXPECT_EQ(control[kDeviceGenerationControlConsumedVerifierRowCount], 6);
    EXPECT_EQ(
        control[kDeviceGenerationControlPublishedStateCommitCount],
        6);
    EXPECT_EQ(control[kDeviceGenerationControlAttemptedDraftTokenCount], 9);
    EXPECT_EQ(control[kDeviceGenerationControlVerifierTokenCount], 12);
    EXPECT_EQ(control[kDeviceGenerationControlLastTransactionDraftDepth], 3);
    EXPECT_EQ(control[kDeviceGenerationControlLastTransactionEmittedTokenCount], 1);

    const ControlRow terminal_control = control;
    const auto terminal_response = response;
    EXPECT_TRUE(append_speculative_outcome_to_device_generation(
        stop_tokens.data(),
        static_cast<int>(stop_tokens.size()),
        stop_meta.data(),
        static_cast<int>(stop_meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));
    EXPECT_EQ(control, terminal_control);
    EXPECT_EQ(response, terminal_response);
}

TEST(Test__DeviceGenerationController, MaintenanceClippingPreservesEmittedConditionCarry)
{
    using namespace llaminar2::sampling_math;

    constexpr int verifier_rows = 4;
    ControlRow control{};
    std::array<int32_t, 16> response{};
    response.fill(-1);
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/8,
        static_cast<int>(response.size()),
        fixedDepthPolicy(verifier_rows - 1),
        control.data()));

    /*
     * Transaction one rejects after consuming token 10.  Its correction token
     * 20 is emitted immediately, but has no published target-model state yet;
     * it must therefore return as the already-committed row zero of the next
     * verifier transaction.
     */
    const std::array<int32_t, verifier_rows> rejection_tokens = {
        10, 20, -1, -1};
    MetaRow rejection_meta = makeMeta(
        /*output_count=*/2,
        /*leading_count=*/0,
        /*verifier_state_count=*/1,
        /*accepted_prefix=*/0,
        /*consumed_rows=*/1,
        /*all_accepted=*/false);
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            verifier_rows,
            verifier_rows,
            control.data()),
        verifier_rows);

    int restore_row = -1;
    int target_cached_tokens = -1;
    int accepted_state_count = -1;
    int publication_ok = 0;
    int32_t next_condition_token = -1;
    ASSERT_TRUE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            rejection_tokens.data(),
            static_cast<int>(rejection_tokens.size()),
            rejection_meta.data(),
            static_cast<int>(rejection_meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/verifier_rows,
            /*base_cached_tokens=*/100,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token));
    ASSERT_EQ(
        control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
        1);

    /*
     * Maintenance permits one newly visible state row.  Because token 20 is a
     * carried row zero, the verifier can also produce token 30.  Token 30 is
     * emitted now but its state is deliberately not published until the next
     * transaction, so the carry bit must remain set.
     */
    std::array<int32_t, verifier_rows> clipped_tokens{};
    clipped_tokens.fill(-1);
    std::array<int, verifier_rows - 1> clipped_samples = {30, 31, 32};
    std::array<int, verifier_rows - 1> clipped_acceptance = {1, 1, 1};
    MetaRow clipped_meta{};
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            verifier_rows,
            /*maintenance_rows_remaining=*/1,
            control.data()),
        1);
    summarize_speculative_verify_batch_at_commit_boundary(
        /*first_token=*/20,
        clipped_samples.data(),
        clipped_acceptance.data(),
        static_cast<int>(clipped_samples.size()),
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        /*bonus_ready_token=*/33,
        /*has_bonus_ready_token=*/1,
        /*max_state_commit_rows=*/1,
        clipped_tokens.data(),
        static_cast<int>(clipped_tokens.size()),
        clipped_meta.data(),
        /*greedy_draft_tokens=*/nullptr,
        /*leading_committed_output_count=*/1);
    ASSERT_EQ(clipped_meta[kSpecBatchMetaOutputCount], 2);
    ASSERT_EQ(clipped_meta[kSpecBatchMetaTargetVerifierStateCommitCount], 2);
    ASSERT_TRUE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            clipped_tokens.data(),
            static_cast<int>(clipped_tokens.size()),
            clipped_meta.data(),
            static_cast<int>(clipped_meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/verifier_rows,
            /*base_cached_tokens=*/101,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token));
    EXPECT_EQ(accepted_state_count, 1);
    EXPECT_EQ(restore_row, 0);
    EXPECT_EQ(next_condition_token, 30);
    EXPECT_EQ(
        control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
        1);
    EXPECT_EQ(
        control[kDeviceGenerationControlPublishedStateCommitCount],
        2);

    /*
     * The next ordinary transaction consumes token 30 as row zero and emits
     * token 40 exactly once.  The explicit response byte sequence is the
     * regression oracle for the production symptom: clearing the carry at the
     * clipped boundary produces {10,20,30,30,40} instead.
     */
    const std::array<int32_t, verifier_rows> continuation_tokens = {
        30, 40, -1, -1};
    MetaRow continuation_meta = makeMeta(
        /*output_count=*/2,
        /*leading_count=*/1,
        /*verifier_state_count=*/1,
        /*accepted_prefix=*/0,
        /*consumed_rows=*/1,
        /*all_accepted=*/false);
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            verifier_rows,
            verifier_rows,
            control.data()),
        verifier_rows);
    ASSERT_TRUE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            continuation_tokens.data(),
            static_cast<int>(continuation_tokens.size()),
            continuation_meta.data(),
            static_cast<int>(continuation_meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/verifier_rows,
            /*base_cached_tokens=*/102,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token));

    const std::array<int32_t, 4> expected = {10, 20, 30, 40};
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 4);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), response.begin()));
}

TEST(Test__DeviceGenerationController, EveryProductionMTPDepthAppendsByteExactly)
{
    using namespace llaminar2::sampling_math;

    for (int mtp_depth = 1; mtp_depth <= 15; ++mtp_depth)
    {
        const int output_count = mtp_depth + 1;
        std::vector<int32_t> compact_tokens(output_count);
        for (int row = 0; row < output_count; ++row)
            compact_tokens[row] = 1000 + mtp_depth * 32 + row;

        const MetaRow meta = makeMeta(
            output_count,
            /*leading_count=*/0,
            /*verifier_state_count=*/mtp_depth,
            /*accepted_prefix=*/mtp_depth,
            /*consumed_rows=*/mtp_depth,
            /*all_accepted=*/true);
        std::vector<int32_t> response(output_count, -1);
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            output_count,
            output_count,
            fixedDepthPolicy(mtp_depth),
            control.data()));
        ASSERT_EQ(
            prepare_device_generation_transaction_budget(
                output_count,
                output_count,
                control.data()),
            output_count);
        ASSERT_TRUE(append_speculative_outcome_to_device_generation(
            compact_tokens.data(),
            output_count,
            meta.data(),
            static_cast<int>(meta.size()),
            response.data(),
            output_count,
            control.data()));

        EXPECT_EQ(response, compact_tokens) << "MTP depth " << mtp_depth;
        EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], output_count);
        EXPECT_EQ(control[kDeviceGenerationControlRemainingTokenCount], 0);
        EXPECT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
        EXPECT_EQ(
            control[kDeviceGenerationControlAcceptedSpeculativeTokenCount],
            mtp_depth);
        EXPECT_EQ(
            control[kDeviceGenerationControlPublishedStateCommitCount],
            mtp_depth);
    }
}

TEST(Test__DeviceGenerationController, ContinuationTokenIsTotalForEveryDepthAndBoundary)
{
    using namespace llaminar2::sampling_math;

    constexpr int32_t first_token = 700;
    constexpr int32_t bonus_token = 900;
    for (int mtp_depth = 1; mtp_depth <= 15; ++mtp_depth)
    {
        std::vector<int32_t> row_tokens(static_cast<size_t>(mtp_depth));
        std::vector<int> row_accepted(static_cast<size_t>(mtp_depth), 1);
        for (int row = 0; row < mtp_depth; ++row)
            row_tokens[static_cast<size_t>(row)] = 1000 + mtp_depth * 32 + row;

        /*
         * Every positive serial-visible boundary has one exact continuation:
         * the first verifier token beyond a clipped boundary, or the sampled
         * bonus token when the complete speculative transaction fits.
         */
        for (int commit_budget = 1;
             commit_budget <= mtp_depth + 1;
             ++commit_budget)
        {
            std::vector<int32_t> compact_tokens(
                static_cast<size_t>(mtp_depth + 1),
                -1);
            MetaRow meta{};
            summarize_speculative_verify_batch_at_commit_boundary(
                first_token,
                row_tokens.data(),
                row_accepted.data(),
                mtp_depth,
                /*stop_tokens=*/nullptr,
                /*stop_token_count=*/0,
                bonus_token,
                /*has_bonus_ready_token=*/1,
                commit_budget,
                compact_tokens.data(),
                static_cast<int>(compact_tokens.size()),
                meta.data());

            ASSERT_EQ(meta[kSpecBatchMetaOk], 1)
                << "depth=" << mtp_depth
                << " budget=" << commit_budget;
            EXPECT_EQ(meta[kSpecBatchMetaOutputCount], commit_budget);
            EXPECT_EQ(
                meta[kSpecBatchMetaTargetVerifierStateCommitCount],
                commit_budget);

            int restore_row = -2;
            int target_cached_tokens = -2;
            int accepted_state_count = -2;
            int publication_ok = 0;
            int32_t next_condition_token = -2;
            int all_drafts_accepted = -2;
            int stopped = -2;
            derive_speculative_publication_metadata(
                meta.data(),
                static_cast<int>(meta.size()),
                /*request_index=*/0,
                /*padded_state_rows_per_request=*/mtp_depth + 1,
                /*base_cached_tokens=*/31,
                commit_budget,
                &restore_row,
                &target_cached_tokens,
                &accepted_state_count,
                &publication_ok,
                compact_tokens.data(),
                static_cast<int>(compact_tokens.size()),
                &next_condition_token,
                &all_drafts_accepted,
                &stopped);

            const int32_t expected_condition =
                commit_budget < mtp_depth + 1
                    ? row_tokens[static_cast<size_t>(commit_budget - 1)]
                    : bonus_token;
            EXPECT_EQ(next_condition_token, expected_condition)
                << "depth=" << mtp_depth
                << " budget=" << commit_budget;
            EXPECT_EQ(publication_ok, 1);
            EXPECT_EQ(accepted_state_count, commit_budget);
            EXPECT_EQ(target_cached_tokens, 31 + commit_budget);
            EXPECT_EQ(restore_row, commit_budget - 1);
            EXPECT_EQ(
                all_drafts_accepted,
                commit_budget == mtp_depth + 1 ? 1 : 0);
            EXPECT_EQ(stopped, 0);
        }

        /*
         * A rejection emits its correction token but cannot publish the state
         * produced by consuming that token.  The correction is therefore both
         * the independently expected next condition and the controller's one
         * carried, already-emitted row for the following transaction.
         */
        for (int rejected_row = 0; rejected_row < mtp_depth; ++rejected_row)
        {
            std::fill(row_accepted.begin(), row_accepted.end(), 1);
            row_accepted[static_cast<size_t>(rejected_row)] = 0;
            std::vector<int32_t> compact_tokens(
                static_cast<size_t>(mtp_depth + 1),
                -1);
            MetaRow meta{};
            summarize_speculative_verify_batch_at_commit_boundary(
                first_token,
                row_tokens.data(),
                row_accepted.data(),
                mtp_depth,
                /*stop_tokens=*/nullptr,
                /*stop_token_count=*/0,
                bonus_token,
                /*has_bonus_ready_token=*/1,
                /*max_state_commit_rows=*/mtp_depth + 1,
                compact_tokens.data(),
                static_cast<int>(compact_tokens.size()),
                meta.data());

            ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
            ASSERT_EQ(meta[kSpecBatchMetaOutputCount], rejected_row + 2);
            ASSERT_EQ(
                meta[kSpecBatchMetaTargetVerifierStateCommitCount],
                rejected_row + 1);

            int restore_row = -2;
            int target_cached_tokens = -2;
            int accepted_state_count = -2;
            int publication_ok = 0;
            int32_t next_condition_token = -2;
            derive_speculative_publication_metadata(
                meta.data(),
                static_cast<int>(meta.size()),
                /*request_index=*/0,
                /*padded_state_rows_per_request=*/mtp_depth + 1,
                /*base_cached_tokens=*/47,
                /*max_state_commit_rows=*/mtp_depth + 1,
                &restore_row,
                &target_cached_tokens,
                &accepted_state_count,
                &publication_ok,
                compact_tokens.data(),
                static_cast<int>(compact_tokens.size()),
                &next_condition_token);
            EXPECT_EQ(
                next_condition_token,
                row_tokens[static_cast<size_t>(rejected_row)])
                << "depth=" << mtp_depth
                << " rejected_row=" << rejected_row;

            ControlRow control{};
            std::vector<int32_t> response(
                static_cast<size_t>(mtp_depth + 1),
                -1);
            ASSERT_TRUE(initialize_device_generation_control(
                mtp_depth + 1,
                static_cast<int>(response.size()),
                fixedDepthPolicy(mtp_depth),
                control.data()));
            ASSERT_EQ(
                prepare_device_generation_transaction_budget(
                    mtp_depth + 1,
                    mtp_depth + 1,
                    control.data()),
                mtp_depth + 1);
            ASSERT_TRUE(append_speculative_outcome_to_device_generation(
                compact_tokens.data(),
                static_cast<int>(compact_tokens.size()),
                meta.data(),
                static_cast<int>(meta.size()),
                response.data(),
                static_cast<int>(response.size()),
                control.data()));
            EXPECT_EQ(
                control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
                1);
        }
    }
}

TEST(Test__DeviceGenerationController,
     AdmissionCarriesAlreadyEmittedRowWithoutDuplicatingResponse)
{
    using namespace llaminar2::sampling_math;

    ControlRow control{};
    std::array<int32_t, 2> response{-1, -1};
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/1,
        static_cast<int>(response.size()),
        fixedDepthPolicy(1),
        control.data(),
        DeviceGenerationLeadingRowDisposition::AlreadyEmitted));
    EXPECT_EQ(
        control[kDeviceGenerationControlNextLeadingCommittedOutputCount],
        1);
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(
            /*verifier_row_capacity=*/2,
            /*maintenance_rows_remaining=*/2,
            control.data()),
        1);

    const std::array<int32_t, 2> compact_tokens{41, 42};
    const MetaRow meta = makeMeta(
        /*output_count=*/2,
        /*leading_count=*/1,
        /*verifier_state_count=*/2,
        /*accepted_prefix=*/1,
        /*consumed_rows=*/1,
        /*all_accepted=*/true);
    ASSERT_TRUE(append_speculative_outcome_to_device_generation(
        compact_tokens.data(),
        static_cast<int>(compact_tokens.size()),
        meta.data(),
        static_cast<int>(meta.size()),
        response.data(),
        static_cast<int>(response.size()),
        control.data()));

    EXPECT_EQ(response[0], 42)
        << "The correction in row zero crossed the preceding response boundary";
    EXPECT_EQ(response[1], -1);
    EXPECT_EQ(control[kDeviceGenerationControlResponseTokenCount], 1);
    EXPECT_EQ(control[kDeviceGenerationControlRemainingTokenCount], 0);
}

TEST(Test__DeviceGenerationController, StopTokensRemainTheTerminalContinuation)
{
    using namespace llaminar2::sampling_math;

    constexpr int mtp_depth = 15;
    constexpr int32_t first_token = 500;
    std::array<int32_t, mtp_depth> row_tokens{};
    std::array<int, mtp_depth> row_accepted{};
    row_accepted.fill(1);
    for (int row = 0; row < mtp_depth; ++row)
        row_tokens[static_cast<size_t>(row)] = 600 + row;

    for (int stop_row = -1; stop_row < mtp_depth; ++stop_row)
    {
        const int32_t stop_token =
            stop_row < 0
                ? first_token
                : row_tokens[static_cast<size_t>(stop_row)];
        std::array<int32_t, mtp_depth + 1> compact_tokens{};
        compact_tokens.fill(-1);
        MetaRow meta{};
        summarize_speculative_verify_batch_at_commit_boundary(
            first_token,
            row_tokens.data(),
            row_accepted.data(),
            mtp_depth,
            &stop_token,
            /*stop_token_count=*/1,
            /*bonus_ready_token=*/999,
            /*has_bonus_ready_token=*/1,
            /*max_state_commit_rows=*/mtp_depth + 1,
            compact_tokens.data(),
            static_cast<int>(compact_tokens.size()),
            meta.data());

        ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
        ASSERT_EQ(meta[kSpecBatchMetaStoppedOnOutput], 1);
        int restore_row = -2;
        int target_cached_tokens = -2;
        int accepted_state_count = -2;
        int publication_ok = 0;
        int32_t next_condition_token = -2;
        int stopped = 0;
        derive_speculative_publication_metadata(
            meta.data(),
            static_cast<int>(meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/mtp_depth + 1,
            /*base_cached_tokens=*/11,
            /*max_state_commit_rows=*/mtp_depth + 1,
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            compact_tokens.data(),
            static_cast<int>(compact_tokens.size()),
            &next_condition_token,
            /*out_all_drafts_accepted=*/nullptr,
            &stopped);
        EXPECT_EQ(next_condition_token, stop_token)
            << "stop_row=" << stop_row;
        EXPECT_EQ(stopped, 1);
    }
}

TEST(Test__DeviceGenerationController, SuccessfulTerminalReplayPublishesOnlyInertState)
{
    using namespace llaminar2::sampling_math;

    ControlRow control{};
    std::array<int32_t, 4> response{};
    response.fill(-1);
    ASSERT_TRUE(initialize_device_generation_control(
        /*max_new_tokens=*/2,
        static_cast<int>(response.size()),
        fixedDepthPolicy(1),
        control.data()));
    ASSERT_EQ(
        prepare_device_generation_transaction_budget(2, 2, control.data()),
        2);

    std::array<int32_t, 2> compact_tokens = {41, 42};
    MetaRow meta = makeMeta(
        /*output_count=*/2,
        /*leading_count=*/0,
        /*verifier_state_count=*/2,
        /*accepted_prefix=*/1,
        /*consumed_rows=*/1,
        /*all_accepted=*/true);
    meta[kSpecBatchMetaReadyToken] = 43;

    int restore_row = -1;
    int target_cached_tokens = -1;
    int accepted_state_count = -1;
    int publication_ok = 0;
    int32_t next_condition_token = -1;
    int all_drafts_accepted = 0;
    int stopped = 0;
    ASSERT_TRUE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            compact_tokens.data(),
            static_cast<int>(compact_tokens.size()),
            meta.data(),
            static_cast<int>(meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/2,
            /*base_cached_tokens=*/17,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token,
            &all_drafts_accepted,
            &stopped));
    ASSERT_EQ(control[kDeviceGenerationControlRequestComplete], 1);
    ASSERT_EQ(next_condition_token, 43);

    const ControlRow terminal_control = control;
    const auto terminal_response = response;
    meta.fill(-777);
    compact_tokens.fill(-999);
    ASSERT_TRUE(
        commit_device_generation_and_derive_speculative_publication_metadata(
            compact_tokens.data(),
            static_cast<int>(compact_tokens.size()),
            meta.data(),
            static_cast<int>(meta.size()),
            /*request_index=*/0,
            /*padded_state_rows_per_request=*/2,
            /*base_cached_tokens=*/19,
            response.data(),
            static_cast<int>(response.size()),
            control.data(),
            &restore_row,
            &target_cached_tokens,
            &accepted_state_count,
            &publication_ok,
            &next_condition_token,
            &all_drafts_accepted,
            &stopped));

    EXPECT_EQ(control, terminal_control);
    EXPECT_EQ(response, terminal_response);
    EXPECT_EQ(restore_row, -1);
    EXPECT_EQ(target_cached_tokens, 19);
    EXPECT_EQ(accepted_state_count, 0);
    EXPECT_EQ(publication_ok, 0);
    EXPECT_EQ(next_condition_token, 43);
    EXPECT_EQ(all_drafts_accepted, 0);
    EXPECT_EQ(stopped, 1);
}

TEST(Test__DeviceGenerationController, InvalidTransitionsFailHardWithoutClipping)
{
    using namespace llaminar2::sampling_math;

    const std::array<int32_t, 4> tokens = {1, 2, 3, 4};
    std::array<int32_t, 4> response{};

    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, fixedDepthPolicy(3), control.data()));
        const MetaRow meta = makeMeta(2, 1, 1, 0, 1, false);
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 4,
            control.data()));
        expectFatal(control, DeviceGenerationError::LeadingCommittedRowMismatch);
    }
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            1, 4, fixedDepthPolicy(3), control.data()));
        const MetaRow meta = makeMeta(2, 0, 1, 0, 1, false);
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 4,
            control.data()));
        expectFatal(control, DeviceGenerationError::ResponseBudgetExceeded);
    }
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, fixedDepthPolicy(3), control.data()));
        const MetaRow meta = makeMeta(3, 0, 2, 1, 2, false);
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 2,
            control.data()));
        expectFatal(control, DeviceGenerationError::ResponseCapacityExceeded);
    }
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, fixedDepthPolicy(3), control.data()));
        MetaRow meta = makeMeta(2, 0, 1, 0, 1, false);
        meta[kSpecBatchMetaOk] = 0;
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 4,
            control.data()));
        expectFatal(control, DeviceGenerationError::InvalidCompactOutcome);
    }
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, fixedDepthPolicy(3), control.data()));
        const MetaRow meta = makeMeta(2, 0, 3, 0, 1, false);
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 4,
            control.data()));
        expectFatal(control, DeviceGenerationError::InvalidVerifierCounts);
    }
    {
        ControlRow control{};
        ASSERT_TRUE(initialize_device_generation_control(
            4, 4, fixedDepthPolicy(3), control.data()));
        control[kDeviceGenerationControlNextLeadingCommittedOutputCount] = 1;
        const MetaRow meta = makeMeta(1, 1, 1, 0, 1, false);
        EXPECT_FALSE(append_speculative_outcome_to_device_generation(
            tokens.data(), 4, meta.data(), meta.size(), response.data(), 4,
            control.data()));
        expectFatal(control, DeviceGenerationError::EmptyTransaction);
    }
}

TEST(Test__DeviceGenerationController, InvalidAdmissionAndMaintenanceFailHard)
{
    using namespace llaminar2::sampling_math;

    ControlRow control;
    control.fill(-1);
    EXPECT_FALSE(initialize_device_generation_control(
        0, 4, fixedDepthPolicy(3), control.data()));
    EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
    EXPECT_EQ(
        control[kDeviceGenerationControlErrorCode],
        static_cast<int>(DeviceGenerationError::InvalidInitialization));

    control.fill(-1);
    EXPECT_FALSE(initialize_device_generation_control(
        4,
        4,
        fixedDepthPolicy(3),
        control.data(),
        static_cast<DeviceGenerationLeadingRowDisposition>(7)));
    EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
    EXPECT_EQ(
        control[kDeviceGenerationControlErrorCode],
        static_cast<int>(DeviceGenerationError::InvalidInitialization));

    ASSERT_TRUE(initialize_device_generation_control(
        4, 4, fixedDepthPolicy(3), control.data()));
    EXPECT_EQ(
        prepare_device_generation_transaction_budget(4, 0, control.data()),
        0);
    EXPECT_EQ(control[kDeviceGenerationControlOk], 0);
    EXPECT_EQ(
        control[kDeviceGenerationControlErrorCode],
        static_cast<int>(DeviceGenerationError::InvalidController));
}
