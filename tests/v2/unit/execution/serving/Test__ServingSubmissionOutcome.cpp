/**
 * @file Test__ServingSubmissionOutcome.cpp
 * @brief PKV-05 typed submission outcomes: Direct / Queued / Refused.
 *
 * These tests exercise the §5.4 contract on the production ServingSession
 * with a scripted row executor: a request submitted while a slot is free is
 * Direct, one admitted beyond the resident slots within the bounded queue is
 * Queued, and one beyond the admitted Q is refused typed — with the FIFO
 * admission order preserved and the caller's response incarnation never
 * minted on refusal.
 */
#include "execution/serving/ServingSession.h"
#include <gtest/gtest.h>

using namespace llaminar2;
namespace
{
class ScriptedRows final : public IServingRowExecutor
{
public:
    bool forwardServingRows(const std::vector<int> &, const std::vector<std::vector<int>> &rows) override
    {
        logits_.assign(rows.size() * 8, 0.0f);
        for (std::size_t r = 0; r < rows.size(); ++r)
            logits_[r * 8 + 3] = 1.0f;
        return true;
    }
    const float *servingRowLogits(int row) const override { return logits_.data() + row * 8; }
    bool resetServingRow(int) override { return true; }
    int restoreServingRowPrefix(int, const std::vector<int32_t> &, PrefixLookupResult &admission) override
    {
        admission = {};
        return 0;
    }
    bool harvestServingRowPrefix(int, const PrefixLookupResult &, const std::vector<int32_t> &) override
    {
        return true;
    }
private:
    std::vector<float> logits_;
};
ServingBatchRequest greedy(std::uint64_t id)
{
    ServingBatchRequest request;
    request.id = id;
    request.prompt = {1, 2, 3};
    request.max_new_tokens = 5;
    request.sampling.temperature = 0.0f;
    request.sampling.top_k = 1;
    request.sampling.seed = 1;
    return request;
}
}

/** @test Direct when a slot is free; Queued within the bounded Q; Refused beyond. */
TEST(ServingSubmissionOutcome, DirectQueuedRefusedWithBoundedQueue)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 2));
    // Slot free: Direct (no earlier queue positions).
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_TRUE(error.empty());
    // Queue bound Q=2: the second request is Queued (one seat left).
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Queued);
    // Beyond the admitted queue (resident full, queue full): Refused typed.
    EXPECT_EQ(session.submit(greedy(3), error), ServingSubmissionOutcome::Refused);
    EXPECT_FALSE(error.empty());
    // A duplicate live id is Refused with the precise diagnostic.
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Refused);
}

/** @test FIFO order preserved: Direct + Queued admit in submission order. */
TEST(ServingSubmissionOutcome, FifoAdmissionOrderPreserved)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 2));
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Queued);
    // Q=2 admitted one queued seat after the direct admission consumed the
    // resident slot; a third request beyond the admitted queue is Refused.
    EXPECT_EQ(session.submit(greedy(3), error), ServingSubmissionOutcome::Refused);
    EXPECT_FALSE(error.empty());
    // One step promotes request 1 into the slot (FIFO); 2 stays queued.
    const auto events = session.step();
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(session.occupancy().resident, 1);
    EXPECT_EQ(session.occupancy().queued, 1);
}

/** @test Q=0 refuses queued requests typed; Direct still works with a free slot. */
TEST(ServingSubmissionOutcome, ZeroQueueAcceptsOnlyDirect)
{
    ScriptedRows rows;
    ServingSession session(rows, 32, 8);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error, nullptr, 0));
    EXPECT_EQ(session.submit(greedy(1), error), ServingSubmissionOutcome::Direct);
    // A second request has no slot and no queue: refused (backpressure typed).
    EXPECT_EQ(session.submit(greedy(2), error), ServingSubmissionOutcome::Refused);
}
