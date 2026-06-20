#include "execution/mtp/MTPSpecRequestBatchScheduler.h"

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

    MTPSpecRequestBatchSchedulerConfig configFor(
        int max_requests = 2,
        int max_draft_tokens = 3,
        MTPSpecRequestBatchMode mode = MTPSpecRequestBatchMode::GREEDY)
    {
        MTPSpecRequestBatchSchedulerConfig config;
        config.max_request_batch = max_requests;
        config.max_draft_tokens = max_draft_tokens;
        config.mode = mode;
        return config;
    }

    MTPSpecSchedulableRequest requestFor(
        int request_id,
        std::vector<int32_t> verifier_tokens,
        std::string key = "qwen36-dense-rocm0",
        int vocab_size = 151936)
    {
        MTPSpecSchedulableRequest request;
        request.request_id = request_id;
        request.compatibility_key = std::move(key);
        request.vocab_size = vocab_size;
        request.base_cached_tokens = 1000 + request_id;
        request.greedy_request.draft_tokens = std::move(verifier_tokens);
        return request;
    }

    std::vector<MTPSpecRequestBatchAdmissionStatus> statuses(
        const MTPSpecRequestBatch &batch)
    {
        std::vector<MTPSpecRequestBatchAdmissionStatus> result;
        result.reserve(batch.admissions.size());
        for (const MTPSpecRequestBatchAdmission &admission : batch.admissions)
            result.push_back(admission.status);
        return result;
    }
} // namespace

TEST(Test__MTPSpecRequestBatchScheduler, BuildsStableGreedyBatchUpToCapacity)
{
    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/2));
    std::vector<MTPSpecSchedulableRequest> pending{
        requestFor(/*request_id=*/7, {11, 12, 13}),
        requestFor(/*request_id=*/8, {21, 22, 23}),
        requestFor(/*request_id=*/9, {31, 32, 33}),
    };

    MTPSpecRequestBatch batch = scheduler.buildNextBatch(pending);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.request_count, 2);
    EXPECT_EQ(batch.shape.max_requests, 2);
    EXPECT_EQ(batch.shape.max_draft_tokens, 3);
    EXPECT_EQ(batch.vocab_size, 151936);
    EXPECT_EQ(batch.compatibility_key, "qwen36-dense-rocm0");
    EXPECT_TRUE(batch.requires_shifted_kv_publication);
    EXPECT_THAT(batch.request_ids, ElementsAre(7, 8));
    EXPECT_THAT(batch.base_cached_tokens, ElementsAre(1007, 1008));
    ASSERT_EQ(batch.greedy_requests.size(), 2u);
    EXPECT_THAT(batch.greedy_requests[0].draft_tokens, ElementsAre(11, 12, 13));
    EXPECT_THAT(batch.greedy_requests[1].draft_tokens, ElementsAre(21, 22, 23));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                    MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                    MTPSpecRequestBatchAdmissionStatus::DEFERRED));
    EXPECT_THAT(batch.admissions[2].reason, HasSubstr("max_request_batch"));
}

TEST(Test__MTPSpecRequestBatchScheduler, DefersNotReadyAndIncompatiblePeers)
{
    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/3));
    std::vector<MTPSpecSchedulableRequest> pending{
        requestFor(/*request_id=*/1, {7, 8}),
        requestFor(/*request_id=*/2, {9, 10}, "qwen36-moe-rocm0"),
        requestFor(/*request_id=*/3, {11, 12}),
        requestFor(/*request_id=*/4, {13, 14}),
    };
    pending[2].ready = false;
    pending[3].vocab_size = 128000;

    MTPSpecRequestBatch batch = scheduler.buildNextBatch(pending);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_THAT(batch.request_ids, ElementsAre(1));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                    MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                    MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                    MTPSpecRequestBatchAdmissionStatus::DEFERRED));
    EXPECT_THAT(batch.admissions[1].reason, HasSubstr("compatibility"));
    EXPECT_THAT(batch.admissions[2].reason, HasSubstr("not ready"));
    EXPECT_THAT(batch.admissions[3].reason, HasSubstr("vocabulary"));
}

TEST(Test__MTPSpecRequestBatchScheduler, AllowsVariableVerifierTokenCountsWithinShape)
{
    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/3));
    std::vector<MTPSpecSchedulableRequest> pending{
        requestFor(/*request_id=*/5, {1}),
        requestFor(/*request_id=*/6, {2, 3}),
        requestFor(/*request_id=*/7, {4, 5, 6}),
    };
    pending[1].requires_shifted_kv_publication = false;
    pending[2].requires_shifted_kv_publication = false;

    MTPSpecRequestBatch batch = scheduler.buildNextBatch(pending);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.request_count, 3);
    EXPECT_EQ(batch.shape.max_draft_tokens, 3);
    ASSERT_EQ(batch.greedy_requests.size(), 3u);
    EXPECT_THAT(batch.greedy_requests[0].draft_tokens, ElementsAre(1));
    EXPECT_THAT(batch.greedy_requests[1].draft_tokens, ElementsAre(2, 3));
    EXPECT_THAT(batch.greedy_requests[2].draft_tokens, ElementsAre(4, 5, 6));
    EXPECT_TRUE(batch.requires_shifted_kv_publication);
}

TEST(Test__MTPSpecRequestBatchScheduler, RejectsInvalidRequestsBeforeTheySeedBatch)
{
    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/2,
                                                    /*max_draft_tokens=*/2));
    std::vector<MTPSpecSchedulableRequest> pending{
        requestFor(/*request_id=*/-1, {1}),
        requestFor(/*request_id=*/2, {}),
        requestFor(/*request_id=*/3, {4, 5, 6}),
        requestFor(/*request_id=*/4, {7, 8}),
    };

    MTPSpecRequestBatch batch = scheduler.buildNextBatch(pending);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_THAT(batch.request_ids, ElementsAre(4));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::REJECTED,
                    MTPSpecRequestBatchAdmissionStatus::REJECTED,
                    MTPSpecRequestBatchAdmissionStatus::REJECTED,
                    MTPSpecRequestBatchAdmissionStatus::ADMITTED));
    EXPECT_THAT(batch.admissions[0].reason, HasSubstr("negative"));
    EXPECT_THAT(batch.admissions[1].reason, HasSubstr("no verifier tokens"));
    EXPECT_THAT(batch.admissions[2].reason, HasSubstr("max_draft_tokens"));
}

TEST(Test__MTPSpecRequestBatchScheduler, AdmitsHomogeneousDeviceTokenRows)
{
    MTPSpecRequestBatchScheduler scheduler(configFor());
    MTPSpecSchedulableRequest first = requestFor(/*request_id=*/10, {1, 2});
    first.verifier_input =
        MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW;
    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 1011;
    second.greedy_request.draft_tokens = {3, 4};

    MTPSpecRequestBatch batch = scheduler.buildNextBatch({first, second});

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.verifier_input,
              MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW);
    EXPECT_THAT(batch.request_ids, ElementsAre(10, 11));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                    MTPSpecRequestBatchAdmissionStatus::ADMITTED));
}

TEST(Test__MTPSpecRequestBatchScheduler, DefersMixedVerifierInputPlacement)
{
    MTPSpecRequestBatchScheduler scheduler(configFor(/*max_requests=*/3));
    MTPSpecSchedulableRequest host = requestFor(/*request_id=*/12, {1, 2});
    MTPSpecSchedulableRequest device = requestFor(/*request_id=*/13, {3, 4});
    device.verifier_input =
        MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW;

    MTPSpecRequestBatch batch = scheduler.buildNextBatch({host, device});

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.verifier_input,
              MTPSpecVerifierInputPlacement::HOST_TOKENS);
    EXPECT_THAT(batch.request_ids, ElementsAre(12));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                    MTPSpecRequestBatchAdmissionStatus::DEFERRED));
    EXPECT_THAT(batch.admissions[1].reason, HasSubstr("verifier input"));
}

TEST(Test__MTPSpecRequestBatchScheduler, SeparatesGreedyAndStochasticBatches)
{
    MTPSpecRequestBatchScheduler scheduler(
        configFor(/*max_requests=*/2,
                  /*max_draft_tokens=*/3,
                  MTPSpecRequestBatchMode::STOCHASTIC));
    MTPSpecSchedulableRequest greedy =
        requestFor(/*request_id=*/20, {1, 2, 3});
    MTPSpecSchedulableRequest stochastic =
        requestFor(/*request_id=*/21, {4, 5, 6});
    stochastic.mode = MTPSpecRequestBatchMode::STOCHASTIC;

    MTPSpecRequestBatch batch =
        scheduler.buildNextBatch({greedy, stochastic});

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.mode, MTPSpecRequestBatchMode::STOCHASTIC);
    EXPECT_THAT(batch.request_ids, ElementsAre(21));
    EXPECT_THAT(
        statuses(batch),
        ElementsAre(MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                    MTPSpecRequestBatchAdmissionStatus::ADMITTED));
    EXPECT_THAT(batch.admissions[0].reason, HasSubstr("greedy"));
}

TEST(Test__MTPSpecRequestBatchScheduler, FailsClearlyForInvalidConfig)
{
    MTPSpecRequestBatchScheduler zero_requests(
        configFor(/*max_requests=*/0, /*max_draft_tokens=*/3));
    EXPECT_FALSE(zero_requests.buildNextBatch({requestFor(1, {1})}).ok);

    MTPSpecRequestBatchScheduler zero_depth(
        configFor(/*max_requests=*/1, /*max_draft_tokens=*/0));
    MTPSpecRequestBatch batch =
        zero_depth.buildNextBatch({requestFor(1, {1})});

    EXPECT_FALSE(batch.ok);
    EXPECT_THAT(batch.error, HasSubstr("max_draft_tokens"));
}

} // namespace llaminar2
