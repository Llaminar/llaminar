#include "execution/mtp/MTPSpecRequestBatchOwner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
namespace
{
    using testing::ElementsAre;
    using testing::HasSubstr;

    MTPSpecRequestBatchSchedulerConfig configFor(int max_requests = 2)
    {
        MTPSpecRequestBatchSchedulerConfig config;
        config.max_request_batch = max_requests;
        config.max_draft_tokens = 3;
        config.mode = MTPSpecRequestBatchMode::GREEDY;
        return config;
    }

    MTPSpecSchedulableRequest requestFor(
        int request_id,
        std::vector<int32_t> tokens,
        std::string key = "qwen36-moe-rocm0")
    {
        MTPSpecSchedulableRequest request;
        request.request_id = request_id;
        request.compatibility_key = std::move(key);
        request.vocab_size = 151936;
        request.base_cached_tokens = 4096 + request_id;
        request.greedy_request.draft_tokens = std::move(tokens);
        return request;
    }
} // namespace

TEST(Test__MTPSpecRequestBatchOwner, ScheduleDoesNotDropRequestsUntilCommit)
{
    MTPSpecRequestBatchOwner owner;
    ASSERT_TRUE(owner.enqueueRequest(requestFor(10, {1, 2, 3})));
    ASSERT_TRUE(owner.enqueueRequest(requestFor(11, {4, 5, 6})));
    ASSERT_TRUE(owner.enqueueRequest(requestFor(12, {7, 8, 9})));

    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/2));
    MTPSpecRequestBatch batch = owner.scheduleNextBatch(scheduler);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_TRUE(owner.hasInFlightBatch());
    EXPECT_EQ(owner.pendingCount(), 3u);
    EXPECT_THAT(batch.request_ids, ElementsAre(10, 11));

    ASSERT_TRUE(owner.commitInFlightBatch());
    EXPECT_FALSE(owner.hasInFlightBatch());
    ASSERT_EQ(owner.pendingCount(), 1u);
    EXPECT_EQ(owner.pendingRequests()[0].request_id, 12);
}

TEST(Test__MTPSpecRequestBatchOwner, ReleaseKeepsRequestsAfterFailedTransaction)
{
    MTPSpecRequestBatchOwner owner;
    ASSERT_TRUE(owner.enqueueRequest(requestFor(20, {1})));
    ASSERT_TRUE(owner.enqueueRequest(requestFor(21, {2})));

    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/2));
    ASSERT_TRUE(owner.scheduleNextBatch(scheduler).ok);
    ASSERT_TRUE(owner.releaseInFlightBatch());

    EXPECT_FALSE(owner.hasInFlightBatch());
    EXPECT_EQ(owner.pendingCount(), 2u);

    MTPSpecRequestBatch retry = owner.scheduleNextBatch(scheduler);
    ASSERT_TRUE(retry.ok) << retry.error;
    EXPECT_THAT(retry.request_ids, ElementsAre(20, 21));
}

TEST(Test__MTPSpecRequestBatchOwner, MutationsHardFailWhileBatchInFlight)
{
    MTPSpecRequestBatchOwner owner;
    ASSERT_TRUE(owner.enqueueRequest(requestFor(30, {1})));

    MTPSpecRequestBatchScheduler scheduler(configFor());
    ASSERT_TRUE(owner.scheduleNextBatch(scheduler).ok);

    std::string error;
    EXPECT_FALSE(owner.enqueueRequest(requestFor(31, {2}), &error));
    EXPECT_THAT(error, HasSubstr("in flight"));

    error.clear();
    EXPECT_FALSE(owner.setRequestReady(30, false, &error));
    EXPECT_THAT(error, HasSubstr("in flight"));

    error.clear();
    EXPECT_FALSE(owner.eraseRequest(30, &error));
    EXPECT_THAT(error, HasSubstr("in flight"));

    error.clear();
    MTPSpecRequestBatch second = owner.scheduleNextBatch(scheduler);
    EXPECT_FALSE(second.ok);
    EXPECT_THAT(second.error, HasSubstr("in-flight"));
}

TEST(Test__MTPSpecRequestBatchOwner, RejectsDuplicateAndUnknownRequests)
{
    MTPSpecRequestBatchOwner owner;
    ASSERT_TRUE(owner.enqueueRequest(requestFor(40, {1})));

    std::string error;
    EXPECT_FALSE(owner.enqueueRequest(requestFor(40, {2}), &error));
    EXPECT_THAT(error, HasSubstr("duplicate"));

    error.clear();
    EXPECT_FALSE(owner.eraseRequest(41, &error));
    EXPECT_THAT(error, HasSubstr("unknown"));

    error.clear();
    EXPECT_FALSE(owner.setRequestReady(41, true, &error));
    EXPECT_THAT(error, HasSubstr("unknown"));

    error.clear();
    EXPECT_FALSE(owner.enqueueRequest(requestFor(-1, {3}), &error));
    EXPECT_THAT(error, HasSubstr("negative"));
}

TEST(Test__MTPSpecRequestBatchOwner, CommitRemovesOnlyAdmittedRequests)
{
    MTPSpecRequestBatchOwner owner;
    ASSERT_TRUE(owner.enqueueRequest(requestFor(50, {1, 2, 3})));
    ASSERT_TRUE(owner.enqueueRequest(
        requestFor(51, {4, 5, 6}, "qwen36-moe-cuda0")));
    ASSERT_TRUE(owner.enqueueRequest(requestFor(52, {7, 8, 9})));

    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/2));
    MTPSpecRequestBatch batch = owner.scheduleNextBatch(scheduler);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_THAT(batch.request_ids, ElementsAre(50, 52));

    ASSERT_TRUE(owner.commitInFlightBatch());
    ASSERT_EQ(owner.pendingCount(), 1u);
    EXPECT_EQ(owner.pendingRequests()[0].request_id, 51);
}

TEST(Test__MTPSpecRequestBatchOwner, SchedulerFailureDoesNotEnterInFlightState)
{
    MTPSpecRequestBatchOwner owner;
    MTPSpecRequestBatchScheduler scheduler(configFor());

    MTPSpecRequestBatch batch = owner.scheduleNextBatch(scheduler);

    EXPECT_FALSE(batch.ok);
    EXPECT_FALSE(owner.hasInFlightBatch());
    EXPECT_THAT(batch.error, HasSubstr("no pending"));
}

} // namespace llaminar2
