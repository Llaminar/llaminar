#include "execution/moe/MoEOverlayDispatchCollective.h"

#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    MoEOverlayDispatchGroup groupForTest()
    {
        MoEOverlayDispatchGroup group;
        group.domain_id = 7;
        group.layer_id = 3;
        group.dispatch_group_id = 11;
        group.participant_count = 2;
        group.participant_index = 1;
        group.owner_participant_index = 0;
        group.executor_participant_index = 1;
        group.stage_sequence = 19;
        group.microbatch_id = 0;
        group.decode_sequence = 23;
        return group;
    }
} // namespace

TEST(Test__MoEOverlayDispatchCollective, GroupIdentityIsCachedAndParticipantScoped)
{
    const auto group = groupForTest();

    EXPECT_TRUE(group.isValid());
    EXPECT_TRUE(group.isParticipant());
    EXPECT_TRUE(group.ownsExecution());
}

TEST(Test__MoEOverlayDispatchCollective, NoOpRequestPreservesDispatchIdentity)
{
    const auto group = groupForTest();
    const auto request = MoEOverlayDispatchRequest::noOp(group, 3, 2);

    EXPECT_EQ(request.kind, MoEOverlayDispatchRequestKind::NoOp);
    EXPECT_EQ(request.group.domain_id, group.domain_id);
    EXPECT_EQ(request.group.dispatch_group_id, group.dispatch_group_id);
    EXPECT_EQ(request.group.stage_sequence, group.stage_sequence);
    EXPECT_EQ(request.layer_idx, 3);
    EXPECT_EQ(request.tier_index, 2);
    EXPECT_FALSE(request.hasRoutedWork());
}

TEST(Test__MoEOverlayDispatchCollective, RoutedRequestUsesBorrowedSparseRows)
{
    const auto group = groupForTest();
    const int selected_rows[] = {0, 4, 9};
    const auto request = MoEOverlayDispatchRequest::routedWork(
        group,
        3,
        2,
        selected_rows,
        3,
        6,
        384,
        nullptr,
        nullptr);

    EXPECT_EQ(request.kind, MoEOverlayDispatchRequestKind::RoutedWork);
    EXPECT_TRUE(request.hasRoutedWork());
    EXPECT_EQ(request.selected_rows, selected_rows);
    EXPECT_EQ(request.selected_row_count, 3u);
    EXPECT_EQ(request.routed_entry_count, 6u);
    EXPECT_EQ(request.transfer_bytes, 384u);
}

TEST(Test__MoEOverlayDispatchCollective, MetricsMergeKeepsHotPathCountersAdditive)
{
    MoEOverlayDispatchMetrics total;
    MoEOverlayDispatchMetrics first;
    first.wait_ns = 10;
    first.no_op_count = 1;
    first.selected_row_count = 2;
    first.transfer_bytes = 64;

    MoEOverlayDispatchMetrics second;
    second.wait_ns = 20;
    second.routed_request_count = 1;
    second.routed_entry_count = 3;
    second.remote_endpoint_work_count = 1;
    second.transfer_bytes = 128;

    total.merge(first);
    total.merge(second);

    EXPECT_EQ(total.wait_ns, 30u);
    EXPECT_EQ(total.no_op_count, 1u);
    EXPECT_EQ(total.routed_request_count, 1u);
    EXPECT_EQ(total.remote_endpoint_work_count, 1u);
    EXPECT_EQ(total.selected_row_count, 2u);
    EXPECT_EQ(total.routed_entry_count, 3u);
    EXPECT_EQ(total.transfer_bytes, 192u);
}

TEST(Test__MoEOverlayDispatchCollective, RendezvousCompletesAfterRoutedAndNoOpParticipantsPublish)
{
    MoEOverlayDispatchRendezvous rendezvous({.participant_count = 2, .slot_count = 4});
    auto group0 = groupForTest();
    group0.participant_index = 0;
    auto group1 = group0;
    group1.participant_index = 1;

    const int selected_rows[] = {1, 4};
    auto routed = MoEOverlayDispatchRequest::routedWork(
        group0,
        3,
        2,
        selected_rows,
        2,
        4,
        256,
        nullptr,
        nullptr);
    auto first = rendezvous.publish(routed);
    ASSERT_TRUE(first.ok) << first.error;
    EXPECT_FALSE(first.collective_complete);
    EXPECT_EQ(first.metrics.routed_request_count, 1u);
    EXPECT_EQ(first.metrics.selected_row_count, 2u);

    auto no_op = MoEOverlayDispatchRequest::noOp(group1, 3, 2);
    auto second = rendezvous.publish(no_op);
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(second.collective_complete);
    EXPECT_EQ(second.metrics.routed_request_count, 1u);
    EXPECT_EQ(second.metrics.no_op_count, 1u);
    EXPECT_EQ(second.metrics.selected_row_count, 2u);
    EXPECT_EQ(second.metrics.routed_entry_count, 4u);
    EXPECT_EQ(second.metrics.transfer_bytes, 256u);
}

TEST(Test__MoEOverlayDispatchCollective, RendezvousRejectsDuplicateParticipantPublish)
{
    MoEOverlayDispatchRendezvous rendezvous({.participant_count = 2, .slot_count = 4});
    auto group = groupForTest();
    group.participant_index = 0;

    auto first = rendezvous.publish(MoEOverlayDispatchRequest::noOp(group, 3, 2));
    ASSERT_TRUE(first.ok) << first.error;

    auto duplicate = rendezvous.publish(MoEOverlayDispatchRequest::noOp(group, 3, 2));
    EXPECT_FALSE(duplicate.ok);
    EXPECT_EQ(duplicate.error_code, 4);
}

TEST(Test__MoEOverlayDispatchCollective, RendezvousCancelCompletesAndReportsCancel)
{
    MoEOverlayDispatchRendezvous rendezvous({.participant_count = 2, .slot_count = 4});
    auto group = groupForTest();
    group.participant_index = 0;

    auto canceled = rendezvous.publish(MoEOverlayDispatchRequest::cancel(group, 3, 2, 77));

    EXPECT_FALSE(canceled.ok);
    EXPECT_TRUE(canceled.collective_complete);
    EXPECT_EQ(canceled.error_code, 77);
    EXPECT_EQ(canceled.metrics.cancel_count, 1u);
}

TEST(Test__MoEOverlayDispatchCollective, LocalRendezvousBackendPublishesWithCanonicalGroup)
{
    MoEOverlayLocalRendezvousBackend backend({.participant_count = 1, .slot_count = 2});
    auto request_group = groupForTest();
    request_group.participant_count = 1;
    request_group.participant_index = 0;
    request_group.executor_participant_index = 0;
    auto canonical_group = request_group;
    canonical_group.dispatch_group_id = 17;

    auto request = MoEOverlayDispatchRequest::noOp(request_group, 3, 2);
    auto result = backend.dispatch(canonical_group, request, nullptr);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.collective_complete);
    EXPECT_EQ(result.group.dispatch_group_id, canonical_group.dispatch_group_id);
    EXPECT_EQ(result.metrics.no_op_count, 1u);
}