#include "execution/moe/MoEOverlayDispatchCollective.h"
#include "execution/moe/MoEOverlayMPIDispatchBackend.h"
#include "../../../mocks/MockMPIContext.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <future>
#include <vector>

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

    class RecordingMPIContext final : public llaminar2::test::MockMPIContext
    {
    public:
        explicit RecordingMPIContext(Config config)
            : MockMPIContext(config)
        {
        }

        void broadcast_int32(int32_t *data, size_t count, int root) const override
        {
            last_root = root;
            last_int32.assign(data, data + count);
            ++broadcast_int32_calls;
        }

        mutable int last_root = -1;
        mutable int broadcast_int32_calls = 0;
        mutable std::vector<int32_t> last_int32;
    };

    class QueuedMPIContext final : public llaminar2::test::MockMPIContext
    {
    public:
        explicit QueuedMPIContext(Config config)
            : MockMPIContext(config)
        {
        }

        void enqueue(const MoEOverlayMPIDispatchHeader &header)
        {
            queued_headers.push_back(header.toWords());
        }

        void broadcast_int32(int32_t *data, size_t count, int root) const override
        {
            last_root = root;
            ++broadcast_int32_calls;
            if (queued_headers.empty())
                return;

            auto words = queued_headers.front();
            queued_headers.pop_front();
            ASSERT_EQ(count, words.size());
            std::copy(words.begin(), words.end(), data);
        }

        mutable int last_root = -1;
        mutable int broadcast_int32_calls = 0;
        mutable std::deque<std::array<int32_t, MoEOverlayMPIDispatchHeader::kWordCount>> queued_headers;
    };
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

TEST(Test__MoEOverlayDispatchCollective, MPIHeaderRoundTripsDispatchIdentity)
{
    MoEOverlayMPIDispatchHeader header;
    header.kind = MoEOverlayMPIMessageKind::RoutedWork;
    header.domain_id = 7;
    header.layer_id = 3;
    header.tier_index = 2;
    header.dispatch_group_id = 11;
    header.participant_count = 2;
    header.owner_participant_index = 0;
    header.executor_participant_index = 1;
    header.stage_sequence = (1ull << 40) + 19ull;
    header.microbatch_id = 4;
    header.decode_sequence = (1ull << 39) + 23ull;
    header.selected_row_count = 5;
    header.routed_entry_count = 8;
    header.transfer_bytes = (1ull << 33) + 384ull;
    header.cancel_reason_code = 77;

    const auto words = header.toWords();
    MoEOverlayMPIDispatchHeader parsed;
    std::string error;

    ASSERT_TRUE(MoEOverlayMPIDispatchHeader::fromWords(words.data(), words.size(), parsed, &error)) << error;
    EXPECT_EQ(parsed.kind, MoEOverlayMPIMessageKind::RoutedWork);
    EXPECT_EQ(parsed.domain_id, header.domain_id);
    EXPECT_EQ(parsed.layer_id, header.layer_id);
    EXPECT_EQ(parsed.tier_index, header.tier_index);
    EXPECT_EQ(parsed.dispatch_group_id, header.dispatch_group_id);
    EXPECT_EQ(parsed.participant_count, header.participant_count);
    EXPECT_EQ(parsed.owner_participant_index, header.owner_participant_index);
    EXPECT_EQ(parsed.executor_participant_index, header.executor_participant_index);
    EXPECT_EQ(parsed.stage_sequence, header.stage_sequence);
    EXPECT_EQ(parsed.microbatch_id, header.microbatch_id);
    EXPECT_EQ(parsed.decode_sequence, header.decode_sequence);
    EXPECT_EQ(parsed.selected_row_count, header.selected_row_count);
    EXPECT_EQ(parsed.routed_entry_count, header.routed_entry_count);
    EXPECT_EQ(parsed.transfer_bytes, header.transfer_bytes);
    EXPECT_EQ(parsed.cancel_reason_code, header.cancel_reason_code);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendNoOpBroadcastsHeaderFromRoot)
{
    auto mpi = std::make_shared<RecordingMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 0, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi, .root_rank = 0});

    auto group = groupForTest();
    group.participant_index = 0;
    auto request = MoEOverlayDispatchRequest::noOp(group, 3, 2);

    auto result = backend.dispatch(group, request, nullptr);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.collective_complete);
    EXPECT_EQ(result.metrics.no_op_count, 1u);
    ASSERT_EQ(mpi->broadcast_int32_calls, 1);
    EXPECT_EQ(mpi->last_root, 0);

    MoEOverlayMPIDispatchHeader sent;
    std::string error;
    ASSERT_TRUE(MoEOverlayMPIDispatchHeader::fromWords(
        mpi->last_int32.data(), mpi->last_int32.size(), sent, &error)) << error;
    EXPECT_EQ(sent.kind, MoEOverlayMPIMessageKind::NoOp);
    EXPECT_EQ(sent.domain_id, group.domain_id);
    EXPECT_EQ(sent.layer_id, group.layer_id);
    EXPECT_EQ(sent.tier_index, request.tier_index);
    EXPECT_EQ(sent.dispatch_group_id, group.dispatch_group_id);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendRoutedWorkBroadcastsRoutedMessage)
{
    auto mpi = std::make_shared<RecordingMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 0, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi, .root_rank = 0});

    auto group = groupForTest();
    group.participant_index = 0;
    const int selected_rows[] = {1, 3, 5};
    auto request = MoEOverlayDispatchRequest::routedWork(
        group, 3, 2, selected_rows, 3, 6, 384, nullptr, nullptr);

    backend.beginForward();
    auto result = backend.dispatch(group, request, nullptr);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.collective_complete);
    EXPECT_FALSE(backend.cancelBroadcastSinceForwardBegin());
    EXPECT_EQ(result.request_kind, MoEOverlayDispatchRequestKind::RoutedWork);
    EXPECT_EQ(result.metrics.routed_request_count, 1u);
    EXPECT_EQ(result.metrics.remote_endpoint_work_count, 1u);
    EXPECT_EQ(result.metrics.selected_row_count, 3u);
    EXPECT_EQ(result.metrics.routed_entry_count, 6u);
    EXPECT_EQ(result.metrics.transfer_bytes, 384u);

    MoEOverlayMPIDispatchHeader sent;
    std::string error;
    ASSERT_TRUE(MoEOverlayMPIDispatchHeader::fromWords(
        mpi->last_int32.data(), mpi->last_int32.size(), sent, &error)) << error;
    EXPECT_EQ(sent.kind, MoEOverlayMPIMessageKind::RoutedWork);
    EXPECT_EQ(sent.domain_id, group.domain_id);
    EXPECT_EQ(sent.layer_id, group.layer_id);
    EXPECT_EQ(sent.tier_index, request.tier_index);
    EXPECT_EQ(sent.dispatch_group_id, group.dispatch_group_id);
    EXPECT_EQ(sent.selected_row_count, 3);
    EXPECT_EQ(sent.routed_entry_count, 6);
    EXPECT_EQ(sent.transfer_bytes, 384u);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendLocalRendezvousBroadcastsOneRoutedEnvelope)
{
    auto mpi = std::make_shared<RecordingMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 0, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi,
                                          .root_rank = 0,
                                          .local_participant_count = 2,
                                          .local_rendezvous_slots = 4});

    auto group0 = groupForTest();
    group0.participant_count = 2;
    group0.participant_index = 0;
    group0.owner_participant_index = 0;
    group0.executor_participant_index = 0;
    auto group1 = group0;
    group1.participant_index = 1;

    const int selected_rows[] = {1, 3, 5};
    auto routed = MoEOverlayDispatchRequest::routedWork(
        group0, 3, 2, selected_rows, 3, 6, 384, nullptr, nullptr);
    auto no_op = MoEOverlayDispatchRequest::noOp(group1, 3, 2);

    auto first = std::async(std::launch::async, [&]() {
        return backend.dispatch(group0, routed, nullptr);
    });
    auto second = backend.dispatch(group1, no_op, nullptr);
    auto first_result = first.get();

    ASSERT_TRUE(first_result.ok) << first_result.error;
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(first_result.collective_complete);
    EXPECT_TRUE(second.collective_complete);
    EXPECT_EQ(mpi->broadcast_int32_calls, 1);
    EXPECT_EQ(first_result.metrics.routed_request_count, 1u);
    EXPECT_EQ(first_result.metrics.no_op_count, 1u);
    EXPECT_EQ(first_result.metrics.remote_endpoint_work_count, 1u);
    EXPECT_EQ(first_result.metrics.selected_row_count, 3u);
    EXPECT_EQ(first_result.metrics.routed_entry_count, 6u);
    EXPECT_EQ(second.metrics.routed_request_count, 1u);
    EXPECT_EQ(second.metrics.no_op_count, 1u);

    MoEOverlayMPIDispatchHeader sent;
    std::string error;
    ASSERT_TRUE(MoEOverlayMPIDispatchHeader::fromWords(
        mpi->last_int32.data(), mpi->last_int32.size(), sent, &error)) << error;
    EXPECT_EQ(sent.kind, MoEOverlayMPIMessageKind::RoutedWork);
    EXPECT_EQ(sent.domain_id, group0.domain_id);
    EXPECT_EQ(sent.layer_id, group0.layer_id);
    EXPECT_EQ(sent.tier_index, routed.tier_index);
    EXPECT_EQ(sent.dispatch_group_id, group0.dispatch_group_id);
    EXPECT_EQ(sent.participant_count, 2);
    EXPECT_EQ(sent.owner_participant_index, 0);
    EXPECT_EQ(sent.executor_participant_index, 0);
    EXPECT_EQ(sent.selected_row_count, 3);
    EXPECT_EQ(sent.routed_entry_count, 6);
    EXPECT_EQ(sent.transfer_bytes, 384u);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendReceiveHeaderParsesEndpointNoOp)
{
    auto mpi = std::make_shared<QueuedMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 1, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi, .root_rank = 0});

    MoEOverlayMPIDispatchHeader header;
    header.kind = MoEOverlayMPIMessageKind::NoOp;
    header.domain_id = 9;
    header.layer_id = 5;
    header.tier_index = 2;
    header.dispatch_group_id = 17;
    header.participant_count = 2;
    header.owner_participant_index = 0;
    header.executor_participant_index = 1;
    header.stage_sequence = 99;
    mpi->enqueue(header);

    MoEOverlayMPIDispatchHeader received;
    std::string error;
    ASSERT_TRUE(backend.receiveHeader(received, &error)) << error;
    EXPECT_EQ(received.kind, MoEOverlayMPIMessageKind::NoOp);
    EXPECT_EQ(received.domain_id, 9);
    EXPECT_EQ(received.layer_id, 5);
    EXPECT_EQ(received.tier_index, 2);
    EXPECT_EQ(received.dispatch_group_id, 17);
    EXPECT_EQ(received.stage_sequence, 99u);
    EXPECT_EQ(mpi->last_root, 0);
    EXPECT_EQ(mpi->broadcast_int32_calls, 1);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendReceiveHeaderRejectsBadVersion)
{
    auto mpi = std::make_shared<QueuedMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 1, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi, .root_rank = 0});

    MoEOverlayMPIDispatchHeader header;
    auto words = header.toWords();
    words[0] = 1234;
    mpi->queued_headers.push_back(words);

    MoEOverlayMPIDispatchHeader received;
    std::string error;
    EXPECT_FALSE(backend.receiveHeader(received, &error));
    EXPECT_NE(error.find("unsupported"), std::string::npos);
}

TEST(Test__MoEOverlayDispatchCollective, MPIBackendSendForwardDoneBroadcastsEndpointMessage)
{
    auto mpi = std::make_shared<RecordingMPIContext>(
        llaminar2::test::MockMPIContext::Config{.rank = 0, .world_size = 2});
    MoEOverlayMPIDispatchBackend backend({.mpi_ctx = mpi, .root_rank = 0});

    auto group = groupForTest();
    group.participant_index = 0;
    ASSERT_TRUE(backend.sendForwardDone(group, 3, 2));

    MoEOverlayMPIDispatchHeader sent;
    std::string error;
    ASSERT_TRUE(MoEOverlayMPIDispatchHeader::fromWords(
        mpi->last_int32.data(), mpi->last_int32.size(), sent, &error)) << error;
    EXPECT_EQ(sent.kind, MoEOverlayMPIMessageKind::ForwardDone);
    EXPECT_EQ(sent.domain_id, group.domain_id);
    EXPECT_EQ(sent.layer_id, group.layer_id);
    EXPECT_EQ(sent.tier_index, 2);
}