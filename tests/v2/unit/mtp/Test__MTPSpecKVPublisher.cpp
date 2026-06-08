#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/mtp/MTPSpecKVPublisher.h"
#include "kernels/IKVCache.h"

#include <tuple>

using namespace llaminar2;
using namespace testing;

namespace
{
    class FakeKVCache final : public IKVCache
    {
    public:
        explicit FakeKVCache(int max_tokens = 64)
            : max_tokens_(max_tokens)
        {
        }

        ActivationPrecision k_precision() const override { return ActivationPrecision::FP32; }
        int get_cached_tokens(int layer, int seq_idx = 0) const override
        {
            (void)layer;
            (void)seq_idx;
            return cached_tokens;
        }
        int max_seq_len() const override { return max_tokens_; }
        int n_layers() const override { return 1; }

        bool truncateSequence(int seq_idx, int cached_tokens_arg, void *stream) override
        {
            truncate_calls.emplace_back(seq_idx, cached_tokens_arg, stream);
            if (!truncate_ok)
                return false;
            cached_tokens = cached_tokens_arg;
            return true;
        }

        bool get_kv(int layer, int seq_idx, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr) override
        {
            (void)layer;
            (void)seq_idx;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return false;
        }

        bool get_kv(int layer, int seq_idx, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const override
        {
            (void)layer;
            (void)seq_idx;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = cached_tokens;
            return false;
        }

        bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) override
        {
            (void)layer;
            (void)seq_idx;
            (void)K;
            (void)V;
            (void)num_tokens;
            return false;
        }

        void clear() override { cached_tokens = 0; }
        void clear_sequence(int layer, int seq_idx) override
        {
            (void)layer;
            (void)seq_idx;
            cached_tokens = 0;
        }
        void clear_layer(int layer) override
        {
            (void)layer;
            cached_tokens = 0;
        }

        int cached_tokens = 0;
        bool truncate_ok = true;
        std::vector<std::tuple<int, int, void *>> truncate_calls;

    private:
        int max_tokens_ = 64;
    };

    MTPSpecStepPlan kvPlan(int base_tokens, int accepted_count, int draft_count = 3)
    {
        MTPSpecStepPlan plan;
        plan.request_id = 77;
        plan.draft_count = draft_count;
        plan.target_rows = draft_count + 1;
        plan.base_cached_tokens = base_tokens;
        plan.accepted_count = accepted_count;
        plan.target_cached_tokens = base_tokens + accepted_count;
        return plan;
    }
} // namespace

TEST(Test__MTPSpecKVPublisher, TruncatesMainAndShiftedMTPCachesToAcceptedPrefix)
{
    FakeKVCache main_cache;
    FakeKVCache mtp0;
    FakeKVCache mtp1;
    int explicit_stream = 0;

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            kvPlan(/*base_tokens=*/10, /*accepted_count=*/2),
            main_cache,
            {&mtp0, &mtp1},
            /*seq_idx=*/0,
            &explicit_stream);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.target_cached_tokens, 12);
    EXPECT_EQ(result.main_truncated_tokens, 12);
    EXPECT_THAT(result.mtp_truncated_tokens, ElementsAre(11, 10));
    EXPECT_THAT(main_cache.truncate_calls, ElementsAre(std::make_tuple(0, 12, &explicit_stream)));
    EXPECT_THAT(mtp0.truncate_calls, ElementsAre(std::make_tuple(0, 11, &explicit_stream)));
    EXPECT_THAT(mtp1.truncate_calls, ElementsAre(std::make_tuple(0, 10, &explicit_stream)));
}

TEST(Test__MTPSpecKVPublisher, ZeroAcceptedTruncatesBackToBase)
{
    FakeKVCache main_cache;
    FakeKVCache mtp0;
    FakeKVCache mtp1;

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            kvPlan(/*base_tokens=*/10, /*accepted_count=*/0),
            main_cache,
            {&mtp0, &mtp1},
            /*seq_idx=*/0,
            /*stream=*/nullptr);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.main_truncated_tokens, 10);
    EXPECT_THAT(result.mtp_truncated_tokens, ElementsAre(9, 8));
}

TEST(Test__MTPSpecKVPublisher, RejectsTargetCachedTokenDrift)
{
    FakeKVCache main_cache;
    MTPSpecStepPlan plan = kvPlan(/*base_tokens=*/10, /*accepted_count=*/2);
    plan.target_cached_tokens = 99;

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            plan,
            main_cache,
            {},
            /*seq_idx=*/0,
            /*stream=*/nullptr);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("drifted"));
    EXPECT_TRUE(main_cache.truncate_calls.empty());
}

TEST(Test__MTPSpecKVPublisher, FailsWhenMainTruncateFailsBeforeMTPCaches)
{
    FakeKVCache main_cache;
    FakeKVCache mtp0;
    main_cache.truncate_ok = false;

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            kvPlan(/*base_tokens=*/3, /*accepted_count=*/1),
            main_cache,
            {&mtp0},
            /*seq_idx=*/0,
            /*stream=*/nullptr);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("main KV"));
    EXPECT_THAT(main_cache.truncate_calls, SizeIs(1));
    EXPECT_TRUE(mtp0.truncate_calls.empty());
}

TEST(Test__MTPSpecKVPublisher, RejectsNullMTPCache)
{
    FakeKVCache main_cache;

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            kvPlan(/*base_tokens=*/3, /*accepted_count=*/1),
            main_cache,
            {nullptr},
            /*seq_idx=*/0,
            /*stream=*/nullptr);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("null MTP KV cache"));
}

TEST(Test__MTPSpecKVPublisher, RejectsCapacityOverflow)
{
    FakeKVCache main_cache(/*max_tokens=*/4);

    MTPSpecKVPublicationResult result =
        publishAcceptedMTPSpecKVState(
            kvPlan(/*base_tokens=*/4, /*accepted_count=*/1),
            main_cache,
            {},
            /*seq_idx=*/0,
            /*stream=*/nullptr);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("exceeds main KV capacity"));
    EXPECT_TRUE(main_cache.truncate_calls.empty());
}
